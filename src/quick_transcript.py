# src/quick_transcript.py

import threading
import time
import subprocess
import platform
from typing import List, Dict, Optional, Callable, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import numpy as np

from volume_segmenter import SegmentInfo
from transcript_manager import TranscriptFileManager
from audio_manager import AudioFileManager

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    """Information about a transcribed segment."""
    segment_info: SegmentInfo
    transcription: str
    timestamp: float
    processing_time: float
    status: str  # 'pending', 'processing', 'completed', 'failed'
    error: Optional[str] = None

@dataclass
class QuickTranscriptResult:
    """Result of quick transcript assembly."""
    transcript_id: str
    transcript_text: str
    segments_count: int
    total_duration: float
    processing_summary: Dict
    file_path: str
    clipboard_copied: bool

class QuickTranscriptAssembler:
    """
    Assembles streaming segments into immediate transcription results.
    
    Provides instant transcript delivery with clipboard integration
    and prepares for background enhancement processing.
    """
    
    def __init__(self, 
                 transcript_manager: TranscriptFileManager,
                 audio_manager: AudioFileManager,
                 auto_clipboard: bool = True):
        """
        Initialize the quick transcript assembler.
        
        Args:
            transcript_manager: TranscriptFileManager for file operations
            audio_manager: AudioFileManager for audio storage
            auto_clipboard: Automatically copy results to clipboard
        """
        self.transcript_manager = transcript_manager
        self.audio_manager = audio_manager
        self.auto_clipboard = auto_clipboard
        
        # State management
        self.lock = threading.RLock()
        self.current_session = None
        self.final_transcript_id = None  # Actual transcript ID after saving
        self.transcription_segments: Dict[str, TranscriptionSegment] = {}
        self.segment_order: List[str] = []
        self.is_processing = False
        
        # Processing tracking
        self.session_start_time = None
        self.processing_stats = {
            'segments_received': 0,
            'segments_completed': 0,
            'segments_failed': 0,
            'total_processing_time': 0.0
        }
        
        # Callbacks
        self.segment_completed_callback: Optional[Callable[[TranscriptionSegment], None]] = None
        self.transcript_ready_callback: Optional[Callable[[QuickTranscriptResult], None]] = None
        
        logger.info("QuickTranscriptAssembler initialized")
    
    def start_session(self, session_id: str) -> None:
        """
        Start a new transcription session.
        
        Args:
            session_id: Unique identifier for the session
        """
        with self.lock:
            self.current_session = session_id
            self.transcription_segments = {}
            self.segment_order = []
            self.is_processing = True
            self.session_start_time = time.time()
            self.processing_stats = {
                'segments_received': 0,
                'segments_completed': 0,
                'segments_failed': 0,
                'total_processing_time': 0.0
            }
            
        logger.info(f"QuickTranscriptAssembler: Started session {session_id}")
    
    def add_segment_for_transcription(self, segment_info: SegmentInfo, audio_data) -> str:
        """
        Add a segment for transcription processing.
        
        Args:
            segment_info: SegmentInfo with timing and metadata
            audio_data: Audio data for the segment
            
        Returns:
            Segment ID for tracking
        """
        if not self.is_processing:
            logger.warning("Not in processing mode")
            return ""
            
        segment_id = f"seg_{segment_info.start_time:.1f}_{segment_info.end_time:.1f}"
        
        with self.lock:
            transcription_segment = TranscriptionSegment(
                segment_info=segment_info,
                transcription="",
                timestamp=time.time(),
                processing_time=0.0,
                status='pending'
            )
            
            self.transcription_segments[segment_id] = transcription_segment
            self.segment_order.append(segment_id)
            self.processing_stats['segments_received'] += 1
            
        logger.debug(f"Added segment for transcription: {segment_id}")
        
        # Start transcription in background thread
        threading.Thread(
            target=self._transcribe_segment,
            args=(segment_id, audio_data),
            daemon=True
        ).start()
        
        return segment_id
    
    def _transcribe_segment(self, segment_id: str, audio_data) -> None:
        """
        Transcribe a single segment using Whisper.
        
        Args:
            segment_id: Segment identifier
            audio_data: Audio data to transcribe
        """
        start_time = time.time()
        
        try:
            with self.lock:
                if segment_id not in self.transcription_segments:
                    return
                self.transcription_segments[segment_id].status = 'processing'
            
            # TODO: Integrate with existing Whisper transcription
            # For now, simulate transcription
            transcription = self._whisper_transcribe(audio_data)
            processing_time = time.time() - start_time
            
            with self.lock:
                if segment_id in self.transcription_segments:
                    segment = self.transcription_segments[segment_id]
                    segment.transcription = transcription
                    segment.processing_time = processing_time
                    segment.status = 'completed'
                    
                    self.processing_stats['segments_completed'] += 1
                    self.processing_stats['total_processing_time'] += processing_time
            
            # Trigger callback
            if self.segment_completed_callback:
                try:
                    self.segment_completed_callback(self.transcription_segments[segment_id])
                except Exception as e:
                    logger.error(f"Error in segment completion callback: {e}")
            
            logger.debug(f"Completed transcription for {segment_id}: {len(transcription)} chars")
            
        except Exception as e:
            logger.error(f"Error transcribing segment {segment_id}: {e}")
            
            with self.lock:
                if segment_id in self.transcription_segments:
                    segment = self.transcription_segments[segment_id]
                    segment.status = 'failed'
                    segment.error = str(e)
                    self.processing_stats['segments_failed'] += 1
    
    def _whisper_transcribe(self, audio_data) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_data: Audio data to transcribe (numpy array, float32, normalized)
            
        Returns:
            Transcribed text
        """
        try:
            # Import Whisper and get model settings
            import whisper
            import os
            
            WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
            WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
            
            # Load model if not already cached
            if not hasattr(self, '_whisper_model') or self._whisper_model is None:
                logger.debug(f"Loading Whisper model: {WHISPER_MODEL}")
                self._whisper_model = whisper.load_model(WHISPER_MODEL)
            
            # Ensure audio_data is float32 numpy array
            import numpy as np
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            elif not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Normalize if needed (should be in [-1, 1] range)
            if audio_data.dtype != np.float32:
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data.astype(np.float32) / 32767.0
                else:
                    audio_data = audio_data.astype(np.float32)
            
            # Check minimum length
            if len(audio_data) < 1600:  # Less than 0.1 seconds at 16kHz
                return ""
            
            # Transcribe with Whisper
            if WHISPER_LANGUAGE and WHISPER_LANGUAGE.lower() != "auto":
                result = self._whisper_model.transcribe(audio_data, fp16=False, language=WHISPER_LANGUAGE)
            else:
                result = self._whisper_model.transcribe(audio_data, fp16=False)
            
            text = result["text"].strip()
            logger.debug(f"Whisper transcribed {len(audio_data)/16000:.1f}s -> '{text[:100]}...'")
            return text
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            duration = len(audio_data) / 16000 if hasattr(audio_data, '__len__') else 0
            return f"[Transcription failed - {duration:.1f}s audio]"
    
    def get_current_transcript(self) -> str:
        """
        Get the current assembled transcript from completed segments.
        
        Returns:
            Current transcript text
        """
        with self.lock:
            transcript_parts = []
            
            for segment_id in self.segment_order:
                segment = self.transcription_segments.get(segment_id)
                if segment and segment.status == 'completed' and segment.transcription.strip():
                    transcript_parts.append(segment.transcription.strip())
            
            return ' '.join(transcript_parts)
    
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """
        Wait for all pending segments to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all segments completed, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                pending_count = sum(1 for s in self.transcription_segments.values() 
                                  if s.status in ['pending', 'processing'])
                
                if pending_count == 0:
                    logger.info("QuickTranscriptAssembler: All segments completed")
                    return True
            
            time.sleep(0.1)
        
        logger.warning(f"QuickTranscriptAssembler: Timeout waiting for completion ({timeout}s)")
        return False
    
    def finalize_transcript(self) -> Optional[QuickTranscriptResult]:
        """
        Finalize the transcript and create the result.
        
        Returns:
            QuickTranscriptResult with complete transcript data
        """
        if not self.current_session:
            logger.error("No active session to finalize")
            return None
        
        with self.lock:
            # Wait a bit for any final segments
            self.wait_for_completion(timeout=10.0)
            
            # Assemble final transcript
            final_transcript = self.get_current_transcript()
            
            if not final_transcript.strip():
                logger.warning("No transcription content available")
                final_transcript = "[No transcription content available]"
            
            # Calculate statistics
            total_duration = 0.0
            completed_segments = [s for s in self.transcription_segments.values() 
                                if s.status == 'completed']
            
            if completed_segments:
                total_duration = max(s.segment_info.end_time for s in completed_segments)
            
            processing_summary = self.processing_stats.copy()
            processing_summary['session_duration'] = time.time() - (self.session_start_time or 0)
            
            # Save transcript file
            try:
                file_path = self._save_transcript_file(final_transcript, processing_summary)
                logger.info(f"Saved quick transcript: {file_path}")
            except Exception as e:
                logger.error(f"Failed to save transcript: {e}")
                file_path = ""
            
            # Copy to clipboard
            clipboard_success = False
            if self.auto_clipboard and final_transcript.strip():
                clipboard_success = self._copy_to_clipboard(final_transcript)
            
            # Create result
            result = QuickTranscriptResult(
                transcript_id=getattr(self, 'final_transcript_id', None) or self.current_session,
                transcript_text=final_transcript,
                segments_count=len(completed_segments),
                total_duration=total_duration,
                processing_summary=processing_summary,
                file_path=file_path,
                clipboard_copied=clipboard_success
            )
            
            # Trigger callback
            if self.transcript_ready_callback:
                try:
                    self.transcript_ready_callback(result)
                except Exception as e:
                    logger.error(f"Error in transcript ready callback: {e}")
            
            self.is_processing = False
            logger.info(f"QuickTranscriptAssembler: Finalized transcript {self.current_session}")
            
            return result
    
    def _save_transcript_file(self, transcript_text: str, processing_summary: Dict) -> str:
        """
        Save transcript to file with metadata.
        
        Args:
            transcript_text: The transcribed text
            processing_summary: Processing statistics
            
        Returns:
            Path to saved file
        """
        # Generate metadata
        metadata = {
            'session_id': self.current_session,
            'timestamp': datetime.now().isoformat(),
            'processing_type': 'quick_streaming',
            'segments_count': processing_summary['segments_completed'],
            'total_duration': processing_summary.get('session_duration', 0),
            'processing_stats': processing_summary
        }
        
        # Save using transcript manager
        file_path, transcript_id = self.transcript_manager.create_new_transcript(
            transcript_text,
            generated_filename="streaming_transcript"
        )
        
        logger.info(f"Transcript saved to {file_path} with ID {transcript_id}")
        
        # Update the stored transcript_id to the actual file ID  
        self.final_transcript_id = transcript_id
        
        return file_path
    
    def _copy_to_clipboard(self, text: str) -> bool:
        """
        Copy text to system clipboard.
        
        Args:
            text: Text to copy
            
        Returns:
            True if successful, False otherwise
        """
        try:
            system = platform.system().lower()
            
            if system == 'darwin':  # macOS
                process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                process.communicate(input=text.encode('utf-8'))
            elif system == 'linux':
                process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                process.communicate(input=text.encode('utf-8'))
            elif system == 'windows':
                process = subprocess.Popen(['clip'], stdin=subprocess.PIPE, shell=True)
                process.communicate(input=text.encode('utf-8'))
            else:
                logger.warning(f"Clipboard copy not supported on {system}")
                return False
            
            logger.info("Successfully copied transcript to clipboard")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            return False
    
    def set_segment_completed_callback(self, callback: Callable[[TranscriptionSegment], None]) -> None:
        """Set callback for when individual segments complete."""
        self.segment_completed_callback = callback
        logger.info("QuickTranscriptAssembler: Segment completion callback set")
    
    def set_transcript_ready_callback(self, callback: Callable[[QuickTranscriptResult], None]) -> None:
        """Set callback for when complete transcript is ready."""
        self.transcript_ready_callback = callback
        logger.info("QuickTranscriptAssembler: Transcript ready callback set")
    
    def get_progress_stats(self) -> Dict:
        """Get current progress statistics."""
        with self.lock:
            stats = self.processing_stats.copy()
            stats['session_id'] = self.current_session
            stats['is_processing'] = self.is_processing
            stats['current_transcript_length'] = len(self.get_current_transcript())
            
            # Calculate completion percentage
            total_segments = stats['segments_received']
            if total_segments > 0:
                completed_ratio = stats['segments_completed'] / total_segments
                stats['completion_percentage'] = int(completed_ratio * 100)
            else:
                stats['completion_percentage'] = 0
            
            return stats

class QuickTranscriptDisplay:
    """Helper for displaying real-time transcription progress."""
    
    @staticmethod
    def format_progress_line(assembler: QuickTranscriptAssembler) -> str:
        """Format a progress line for display."""
        stats = assembler.get_progress_stats()
        
        if not stats['is_processing']:
            return "📝 Transcription ready"
        
        completed = stats['segments_completed']
        total = stats['segments_received']
        percentage = stats['completion_percentage']
        
        if total == 0:
            return "📝 Waiting for audio segments..."
        
        return f"📝 Transcribing: {completed}/{total} segments ({percentage}%)"
    
    @staticmethod
    def format_segment_update(segment: TranscriptionSegment) -> str:
        """Format a segment completion update."""
        duration = segment.segment_info.duration
        text_preview = segment.transcription[:50] + "..." if len(segment.transcription) > 50 else segment.transcription
        
        return f"📝 Segment {duration:.1f}s: \"{text_preview}\""
    
    @staticmethod
    def format_final_result(result: QuickTranscriptResult) -> List[str]:
        """Format the final result announcement."""
        lines = [
            "✅ Recording complete!",
            f"📝 Transcript: {len(result.transcript_text)} characters",
            f"⏱️  Duration: {result.total_duration:.1f} seconds",
            f"🔢 Segments: {result.segments_count}"
        ]
        
        if result.clipboard_copied:
            lines.append("📋 Transcript copied to clipboard")
        
        if result.file_path:
            lines.append(f"📁 Saved: {os.path.basename(result.file_path)}")
        
        return lines