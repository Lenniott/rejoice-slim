# src/background_enhancer.py

import threading
import time
import os
import tempfile
import wave
import numpy as np
import whisper
from typing import Optional, Dict, Callable
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from transcript_manager import TranscriptFileManager
from audio_manager import AudioFileManager
from summarization_service import SummarizationService
from quick_transcript import QuickTranscriptResult

logger = logging.getLogger(__name__)

@dataclass
class EnhancementTask:
    """Background enhancement task information."""
    session_id: str
    master_audio_path: str
    quick_transcript_path: str
    quick_transcript_text: str
    start_time: float
    status: str  # 'pending', 'transcribing', 'summarizing', 'completed', 'failed'
    progress_percentage: int = 0
    enhanced_transcript: Optional[str] = None
    enhanced_summary: Optional[bool] = None
    error_message: Optional[str] = None
    completion_time: Optional[float] = None

class BackgroundEnhancer:
    """
    Silent quality improvement worker for background processing.
    
    Processes full audio files with Whisper for maximum accuracy
    and generates enhanced summaries without user intervention.
    """
    
    def __init__(self, 
                 transcript_manager: TranscriptFileManager,
                 audio_manager: AudioFileManager,
                 summarization_service: SummarizationService,
                 whisper_model_name: str = "base",
                 whisper_language: str = "auto",
                 sample_rate: int = 16000,
                 auto_cleanup: bool = True,
                 apply_metadata: bool = True):
        """
        Initialize the background enhancer.
        
        Args:
            transcript_manager: TranscriptFileManager for file operations
            audio_manager: AudioFileManager for audio storage
            summarization_service: SummarizationService for summary generation
            whisper_model_name: Whisper model to use for transcription
            whisper_language: Language for transcription ('auto' for detection)
            sample_rate: Audio sample rate (default 16000 for Whisper)
            auto_cleanup: Automatically cleanup audio files after successful enhancement
            apply_metadata: Generate AI metadata (summary, tags) after enhancement
        """
        self.transcript_manager = transcript_manager
        self.audio_manager = audio_manager
        self.summarization_service = summarization_service
        self.whisper_model_name = whisper_model_name
        self.whisper_language = whisper_language
        self.sample_rate = sample_rate
        self.auto_cleanup = auto_cleanup
        self.apply_metadata = apply_metadata
        
        # Lazy-load Whisper model (loaded in worker thread)
        self._whisper_model = None
        
        # Task management
        self.lock = threading.RLock()
        self.enhancement_queue = {}  # session_id -> EnhancementTask
        self.worker_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'audio_files_cleaned': 0
        }
        
        # Callbacks
        self.task_completed_callback: Optional[Callable[[EnhancementTask], None]] = None
        self.progress_callback: Optional[Callable[[str, int], None]] = None
        
        logger.info(
            f"BackgroundEnhancer initialized (auto-cleanup: {auto_cleanup}, metadata: {apply_metadata})"
        )
    
    def _load_whisper_model(self):
        """Lazy-load Whisper model in worker thread."""
        if self._whisper_model is None:
            logger.info(f"Loading Whisper model '{self.whisper_model_name}' in background...")
            self._whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info("Whisper model loaded in background")
        return self._whisper_model
    
    def start_worker(self) -> None:
        """Start the background worker thread."""
        with self.lock:
            if self.is_running:
                logger.warning("Background enhancer already running")
                return
            
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            
        logger.info("BackgroundEnhancer: Worker thread started")
    
    def stop_worker(self) -> None:
        """Stop the background worker thread."""
        with self.lock:
            self.is_running = False
            
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            
        logger.info("BackgroundEnhancer: Worker thread stopped")
    
    def queue_enhancement(self, 
                         quick_result: QuickTranscriptResult, 
                         master_audio_path: str) -> bool:
        """
        Queue a transcript for background enhancement.
        
        Args:
            quick_result: QuickTranscriptResult from immediate processing
            master_audio_path: Path to master audio file
            
        Returns:
            True if queued successfully
        """
        if not quick_result.has_content:
            logger.info("BackgroundEnhancer: Skipping enhancement for empty quick transcript")
            return False
        
        if not os.path.exists(master_audio_path):
            logger.error(f"Master audio file not found: {master_audio_path}")
            return False
        
        task = EnhancementTask(
            session_id=quick_result.transcript_id,
            master_audio_path=master_audio_path,
            quick_transcript_path=quick_result.file_path,
            quick_transcript_text=quick_result.transcript_text,
            start_time=time.time(),
            status='pending'
        )
        
        with self.lock:
            self.enhancement_queue[task.session_id] = task
            
        logger.info(f"Queued enhancement for session {task.session_id}")
        return True
    
    def _worker_loop(self) -> None:
        """Main worker loop for background processing."""
        logger.debug("BackgroundEnhancer: Worker loop started")
        
        while self.is_running:
            try:
                # Look for pending tasks
                task = self._get_next_task()
                
                if task is None:
                    time.sleep(1.0)  # No work to do
                    continue
                
                # Process the task
                self._process_enhancement_task(task)
                
            except Exception as e:
                logger.error(f"Error in background enhancement worker loop: {e}")
                time.sleep(5.0)  # Wait before retrying
        
        logger.debug("BackgroundEnhancer: Worker loop ended")
    
    def _get_next_task(self) -> Optional[EnhancementTask]:
        """Get the next pending task from the queue."""
        with self.lock:
            for task in self.enhancement_queue.values():
                if task.status == 'pending':
                    task.status = 'transcribing'
                    return task
        return None
    
    def _process_enhancement_task(self, task: EnhancementTask) -> None:
        """
        Process a complete enhancement task.
        
        Args:
            task: EnhancementTask to process
        """
        logger.info(f"Starting background enhancement for {task.session_id}")
        
        try:
            # Step 1: Enhanced transcription with full audio
            self._update_progress(task, 10, "Starting full-audio transcription...")
            enhanced_transcript = self._enhance_transcription(task)
            
            if not enhanced_transcript:
                raise Exception("Failed to generate enhanced transcription")
            
            task.enhanced_transcript = enhanced_transcript
            self._update_progress(task, 60, "Full-audio transcription complete")
            
            # Step 2: Update transcript file with enhanced content
            self._update_progress(task, 70, "Updating transcript with full audio...")
            if not self._update_transcript_file(task):
                raise Exception("Failed to update transcript with enhanced content")
            self._update_progress(task, 80, "Transcript updated with enhanced audio")
            
            # Step 3: Generate AI metadata if enabled
            if self.apply_metadata and self.summarization_service:
                self._update_progress(task, 90, "Generating AI metadata from enhanced transcript...")
                try:
                    metadata_success = self.summarization_service.summarize_file(
                        task.quick_transcript_path,
                        copy_to_notes=False
                    )
                    if metadata_success:
                        task.enhanced_summary = True
                        # File may have been renamed by metadata step; refresh path
                        try:
                            new_path = self.transcript_manager.find_transcript(task.session_id)
                            if new_path:
                                task.quick_transcript_path = new_path
                        except Exception:
                            pass
                        self._update_progress(task, 95, "AI metadata updated")
                    else:
                        self._update_progress(task, 95, "AI metadata skipped (failed)")
                        logger.warning(f"AI metadata generation failed for {task.session_id}")
                except Exception as metadata_error:
                    self._update_progress(task, 95, "AI metadata skipped (error)")
                    logger.error(f"Error generating AI metadata for {task.session_id}: {metadata_error}")
            else:
                self._update_progress(task, 90, "AI metadata disabled - skipping")
            
            # Step 4: Cleanup if enabled
            if self.auto_cleanup:
                self._update_progress(task, 98, "Cleaning up audio files...")
                self._cleanup_audio_file(task)
            
            # Mark as completed
            task.status = 'completed'
            task.completion_time = time.time()
            self._update_progress(task, 100, "Enhancement complete")
            
            # Update statistics
            with self.lock:
                self.stats['tasks_completed'] += 1
                self.stats['total_processing_time'] += (task.completion_time - task.start_time)
            
            # Trigger completion callback
            if self.task_completed_callback:
                try:
                    self.task_completed_callback(task)
                except Exception as e:
                    logger.error(f"Error in task completion callback: {e}")
            
            logger.info(f"Completed background enhancement for {task.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to process enhancement task {task.session_id}: {e}")
            
            task.status = 'failed'
            task.error_message = str(e)
            task.completion_time = time.time()
            
            with self.lock:
                self.stats['tasks_failed'] += 1
    
    def _enhance_transcription(self, task: EnhancementTask) -> Optional[str]:
        """
        Generate enhanced transcription using full audio file.
        
        Args:
            task: EnhancementTask with audio file information
            
        Returns:
            Enhanced transcript text or None if failed
        """
        try:
            # Load Whisper model (lazy loading in background thread)
            whisper_model = self._load_whisper_model()
            
            # Read WAV file
            with wave.open(task.master_audio_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Convert to float32 normalized to [-1, 1]
            audio_data = audio_data.astype(np.float32) / 32767.0
            
            # Handle stereo to mono conversion if needed
            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
            
            if len(audio_data) < 1600:  # Less than 0.1 seconds
                logger.warning(f"Audio too short for transcription: {len(audio_data)} samples")
                return task.quick_transcript_text  # Return quick transcript as fallback
            
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != self.sample_rate:
                ratio = self.sample_rate / sample_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Transcribe with Whisper
            if self.whisper_language and self.whisper_language.lower() != "auto":
                result = whisper_model.transcribe(audio_data, fp16=False, language=self.whisper_language)
            else:
                result = whisper_model.transcribe(audio_data, fp16=False)
            
            enhanced_text = result["text"].strip()
            
            if not enhanced_text:
                logger.warning(f"Enhanced transcription empty for {task.session_id}")
                return task.quick_transcript_text
            
            logger.info(f"Enhanced transcription completed for {task.session_id}: {len(enhanced_text)} chars")
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Failed to enhance transcription for {task.session_id}: {e}")
            # Return quick transcript as fallback
            return task.quick_transcript_text
    
    def _update_transcript_file(self, task: EnhancementTask) -> bool:
        """
        Update the transcript file with enhanced content.
        
        Args:
            task: EnhancementTask with enhanced content
        """
        try:
            success = self.transcript_manager.update_transcript_content(
                task.quick_transcript_path,
                task.enhanced_transcript or ""
            )
            if not success:
                raise RuntimeError("Transcript manager reported failure")
            logger.debug(f"Updated transcript file for {task.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update transcript file for {task.session_id}: {e}")
            return False
    
    def _cleanup_audio_file(self, task: EnhancementTask) -> None:
        """
        Clean up audio file after successful enhancement.
        
        Args:
            task: Completed EnhancementTask
        """
        try:
            if os.path.exists(task.master_audio_path):
                os.remove(task.master_audio_path)
                
                with self.lock:
                    self.stats['audio_files_cleaned'] += 1
                
                logger.debug(f"Cleaned up audio file: {task.master_audio_path}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup audio file {task.master_audio_path}: {e}")
    
    def _update_progress(self, task: EnhancementTask, percentage: int, message: str) -> None:
        """
        Update task progress and trigger callbacks.
        
        Args:
            task: EnhancementTask to update
            percentage: Completion percentage (0-100)
            message: Progress message
        """
        task.progress_percentage = percentage
        
        if self.progress_callback:
            try:
                self.progress_callback(task.session_id, percentage)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
        
        logger.debug(f"Enhancement progress for {task.session_id}: {percentage}% - {message}")
    
    def get_task_status(self, session_id: str) -> Optional[Dict]:
        """
        Get status of a specific enhancement task.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            Task status dictionary or None if not found
        """
        with self.lock:
            task = self.enhancement_queue.get(session_id)
            
            if not task:
                return None
            
            return {
                'session_id': task.session_id,
                'status': task.status,
                'progress_percentage': task.progress_percentage,
                'start_time': task.start_time,
                'completion_time': task.completion_time,
                'processing_duration': (
                    (task.completion_time or time.time()) - task.start_time
                ),
                'error_message': task.error_message,
                'has_enhanced_transcript': task.enhanced_transcript is not None,
                'has_enhanced_summary': task.enhanced_summary is not None
            }
    
    def get_queue_stats(self) -> Dict:
        """Get statistics about the enhancement queue."""
        with self.lock:
            queue_stats = {
                'pending': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0
            }
            
            for task in self.enhancement_queue.values():
                if task.status == 'pending':
                    queue_stats['pending'] += 1
                elif task.status in ['transcribing', 'summarizing']:
                    queue_stats['processing'] += 1
                elif task.status == 'completed':
                    queue_stats['completed'] += 1
                elif task.status == 'failed':
                    queue_stats['failed'] += 1
            
            return {
                'is_running': self.is_running,
                'queue': queue_stats,
                'total_tasks': len(self.enhancement_queue),
                'lifetime_stats': self.stats.copy()
            }
    
    def set_task_completed_callback(self, callback: Callable[[EnhancementTask], None]) -> None:
        """Set callback for when enhancement tasks complete."""
        self.task_completed_callback = callback
        logger.info("BackgroundEnhancer: Task completion callback set")
    
    def set_progress_callback(self, callback: Callable[[str, int], None]) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback
        logger.info("BackgroundEnhancer: Progress callback set")
    
    def clear_completed_tasks(self, older_than_hours: int = 24) -> int:
        """
        Clear completed tasks older than specified hours.
        
        Args:
            older_than_hours: Clear tasks completed more than this many hours ago
            
        Returns:
            Number of tasks cleared
        """
        cutoff_time = time.time() - (older_than_hours * 3600)
        cleared_count = 0
        
        with self.lock:
            to_remove = []
            
            for session_id, task in self.enhancement_queue.items():
                if (task.status in ['completed', 'failed'] and 
                    task.completion_time and 
                    task.completion_time < cutoff_time):
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                del self.enhancement_queue[session_id]
                cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} completed enhancement tasks")
        return cleared_count

class BackgroundEnhancementManager:
    """
    Manager for coordinating background enhancement with the main application.
    
    Provides a simple interface for starting background enhancement
    and checking progress without blocking the main application.
    """
    
    def __init__(self, background_enhancer: BackgroundEnhancer):
        """Initialize with a BackgroundEnhancer instance."""
        self.enhancer = background_enhancer
        self.enhancer.start_worker()
        
    def start_enhancement(self, 
                         quick_result: QuickTranscriptResult, 
                         master_audio_path: str) -> bool:
        """
        Start background enhancement for a completed recording.
        
        Args:
            quick_result: Quick transcript result
            master_audio_path: Path to master audio file
            
        Returns:
            True if enhancement started successfully
        """
        success = self.enhancer.queue_enhancement(quick_result, master_audio_path)
        
        if success:
            logger.info(f"🔄 Background enhancement starting for {quick_result.transcript_id}")
        
        return success
    
    def get_enhancement_status(self, session_id: str) -> Optional[str]:
        """
        Get simple status message for an enhancement task.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            Status message or None if not found
        """
        status = self.enhancer.get_task_status(session_id)
        
        if not status:
            return None
        
        status_map = {
            'pending': '⏳ Queued for enhancement',
            'transcribing': f'🎯 Processing audio ({status["progress_percentage"]}%)',
            'summarizing': f'📊 Generating summary ({status["progress_percentage"]}%)',
            'completed': '✅ Enhancement complete',
            'failed': f'❌ Enhancement failed: {status["error_message"]}'
        }
        
        return status_map.get(status['status'], f'Unknown status: {status["status"]}')
    
    def cleanup_and_stop(self) -> None:
        """Clean up and stop the background enhancer."""
        self.enhancer.clear_completed_tasks(older_than_hours=1)
        self.enhancer.stop_worker()