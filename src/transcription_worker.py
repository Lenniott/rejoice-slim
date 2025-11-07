# src/transcription_worker.py

import time
import threading
import queue
import logging
import sys
import os
from contextlib import contextmanager
from typing import List, Optional, Callable, Tuple
import numpy as np
import whisper

logger = logging.getLogger(__name__)

@contextmanager
def suppress_whisper_output():
    """Suppress Whisper's progress bars and verbose output"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class TranscriptionWorker:
    """
    Worker thread that processes audio chunks for transcription.
    
    Handles retry logic, error recovery, and thread-safe result collection.
    Designed to work with a queue-based producer-consumer pattern.
    """
    
    def __init__(self, 
                 worker_id: int,
                 whisper_model: whisper.Whisper,
                 whisper_language: Optional[str] = None,
                 max_retry_attempts: int = 3,
                 retry_delay_base: float = 1.0):
        """
        Initialize the transcription worker.
        
        Args:
            worker_id: Unique identifier for this worker
            whisper_model: Loaded Whisper model instance
            whisper_language: Language code for Whisper (None for auto)
            max_retry_attempts: Maximum retry attempts for failed chunks
            retry_delay_base: Base delay for exponential backoff (seconds)
        """
        self.worker_id = worker_id
        self.whisper_model = whisper_model
        self.whisper_language = whisper_language
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay_base = retry_delay_base
        
        # Threading control
        self.stop_event = threading.Event()
        self.thread = None
        
        # Statistics
        self.chunks_processed = 0
        self.chunks_failed = 0
        self.total_processing_time = 0.0
        
        logger.info(f"TranscriptionWorker {worker_id} initialized")
    
    def start(self, 
              chunk_queue: queue.Queue,
              result_callback: Callable[[str, str, float], None]) -> None:
        """
        Start the worker thread.
        
        Args:
            chunk_queue: Queue containing audio chunks to process
            result_callback: Function to call with transcription results
                            Signature: callback(chunk_id, transcribed_text, timestamp)
        """
        self.chunk_queue = chunk_queue
        self.result_callback = result_callback
        self.stop_event.clear()
        
        self.thread = threading.Thread(
            target=self._worker_loop,
            name=f"TranscriptionWorker-{self.worker_id}",
            daemon=True
        )
        self.thread.start()
        
        logger.info(f"TranscriptionWorker {self.worker_id} started")
    
    def stop(self) -> None:
        """Stop the worker thread gracefully with timeout protection."""
        logger.debug(f"Stopping worker {self.worker_id}")
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)  # Reduced timeout to prevent hanging
            if self.thread.is_alive():
                logger.warning(f"TranscriptionWorker {self.worker_id} didn't stop within timeout")
            else:
                logger.debug(f"TranscriptionWorker {self.worker_id} stopped successfully")
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes chunks from the queue."""
        logger.debug(f"Worker {self.worker_id} entering main loop")
        
        while not self.stop_event.is_set():
            try:
                # Get chunk with timeout to allow checking stop_event
                chunk_data = self.chunk_queue.get(timeout=1.0)
                
                if chunk_data is None:  # Sentinel value to stop
                    break
                
                # Handle both old format (chunk_number, audio) and new format (chunk_info dict)
                if isinstance(chunk_data, dict):
                    chunk_id = chunk_data['id']
                    audio_chunk = chunk_data['data']
                    timestamp = chunk_data['timestamp']
                    retry_count = chunk_data.get('retry_count', 0)
                else:
                    # Legacy format support
                    chunk_number, audio_chunk = chunk_data
                    chunk_id = str(chunk_number)
                    timestamp = time.time()
                    retry_count = 0
                
                self._process_chunk(audio_chunk, chunk_id, timestamp, retry_count)
                
            except queue.Empty:
                # Timeout - check if we should stop
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} unexpected error: {e}")
                break
        
        logger.debug(f"Worker {self.worker_id} exiting main loop")
    
    def _process_chunk(self, audio_chunk: np.ndarray, chunk_id: str, timestamp: float, retry_count: int = 0) -> None:
        """
        Process a single audio chunk with retry logic.
        
        Args:
            audio_chunk: Audio data as numpy array
            chunk_id: Unique identifier for this chunk
            timestamp: Timestamp when chunk was recorded
            retry_count: Current retry attempt number
        """
        start_time = time.time()
        
        # Validate audio chunk before processing
        if audio_chunk is None or len(audio_chunk) == 0:
            logger.warning(f"Worker {self.worker_id} chunk {chunk_id} is empty, skipping")
            self.result_callback(chunk_id, "", timestamp)
            return
        
        # Check if chunk is too short (less than 1 second)
        duration = len(audio_chunk) / 16000  # Assuming 16kHz sample rate
        if duration < 1.0:
            logger.warning(f"Worker {self.worker_id} chunk {chunk_id} too short ({duration:.1f}s), skipping")
            self.result_callback(chunk_id, "", timestamp)
            return
        
        # Check for silent audio (all zeros or very low amplitude)
        max_amplitude = np.max(np.abs(audio_chunk))
        if max_amplitude < 0.0001:  # More lenient threshold for quiet audio
            logger.warning(f"Worker {self.worker_id} chunk {chunk_id} appears silent (max amp: {max_amplitude:.6f})")
            self.result_callback(chunk_id, "", timestamp)
            return
        
        logger.debug(f"Worker {self.worker_id} processing chunk {chunk_id}: {duration:.1f}s, max_amp: {max_amplitude:.3f}")
        
        for attempt in range(self.max_retry_attempts + 1):
            try:
                # Ensure audio is in correct format for Whisper
                if audio_chunk.dtype != np.float32:
                    audio_chunk = audio_chunk.astype(np.float32)
                
                # Normalize audio to [-1, 1] range if needed
                if np.max(np.abs(audio_chunk)) > 1.0:
                    audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
                
                # Transcribe the chunk with suppressed output
                with suppress_whisper_output():
                    if self.whisper_language and self.whisper_language.lower() != "auto":
                        result = self.whisper_model.transcribe(
                            audio_chunk, 
                            fp16=False, 
                            language=self.whisper_language,
                            verbose=False  # Reduce Whisper's own logging
                        )
                    else:
                        result = self.whisper_model.transcribe(
                            audio_chunk, 
                            fp16=False,
                            verbose=False
                        )
                
                transcribed_text = result["text"].strip()
                
                # Success - send result
                self.result_callback(chunk_id, transcribed_text, timestamp)
                
                if transcribed_text:
                    logger.info(f"Worker {self.worker_id} chunk {chunk_id} SUCCESS: '{transcribed_text[:50]}...'")
                else:
                    logger.debug(f"Worker {self.worker_id} chunk {chunk_id} empty result (no speech detected)")
                
                # Update statistics
                self.chunks_processed += 1
                self.total_processing_time += time.time() - start_time
                return
                
            except Exception as e:
                if attempt < self.max_retry_attempts:
                    # Retry with exponential backoff
                    delay = self.retry_delay_base * (2 ** attempt)
                    logger.warning(f"Worker {self.worker_id} chunk {chunk_id} "
                                 f"attempt {attempt + 1}/{self.max_retry_attempts + 1} failed: {e}. "
                                 f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # All retries exhausted
                    self.result_callback(chunk_id, "", timestamp)  # Send empty instead of error text
                    self.chunks_failed += 1
                    
                    logger.error(f"Worker {self.worker_id} chunk {chunk_id} "
                               f"FAILED after {self.max_retry_attempts + 1} attempts: {e}")
                    return
    
    def get_stats(self) -> dict:
        """
        Get worker statistics.
        
        Returns:
            dict: Statistics including processed chunks, failures, etc.
        """
        avg_processing_time = (
            self.total_processing_time / self.chunks_processed 
            if self.chunks_processed > 0 else 0.0
        )
        
        return {
            'worker_id': self.worker_id,
            'chunks_processed': self.chunks_processed,
            'chunks_failed': self.chunks_failed,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': avg_processing_time,
            'success_rate': (
                self.chunks_processed / (self.chunks_processed + self.chunks_failed)
                if (self.chunks_processed + self.chunks_failed) > 0 else 0.0
            )
        }


class TranscriptionWorkerPool:
    """
    Manages a pool of transcription workers for concurrent processing.
    Enhanced for streaming transcription with real-time results.
    """
    
    def __init__(self, 
                 whisper_model: whisper.Whisper,
                 whisper_language: Optional[str] = None,
                 num_workers: int = 2,
                 max_retry_attempts: int = 3):
        """
        Initialize the worker pool.
        
        Args:
            whisper_model: Loaded Whisper model instance
            whisper_language: Language code for Whisper
            num_workers: Number of worker threads to create
            max_retry_attempts: Maximum retry attempts per chunk
        """
        self.whisper_model = whisper_model
        self.whisper_language = whisper_language
        self.num_workers = num_workers
        self.max_retry_attempts = max_retry_attempts
        
        # Threading components
        self.chunk_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        
        # Result collection (enhanced for streaming)
        self.transcribed_segments = []
        self.segments_lock = threading.Lock()
        self.chunk_count = 0
        
        # Streaming results tracking
        self.streaming_results = {}  # chunk_id -> (timestamp, text)
        self.streaming_lock = threading.Lock()
        
        logger.info(f"TranscriptionWorkerPool initialized with {num_workers} workers")
    
    def start(self) -> None:
        """Start all workers in the pool."""
        # Create and start workers
        for i in range(self.num_workers):
            worker = TranscriptionWorker(
                worker_id=i,
                whisper_model=self.whisper_model,
                whisper_language=self.whisper_language,
                max_retry_attempts=self.max_retry_attempts
            )
            worker.start(self.chunk_queue, self._result_callback)
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} transcription workers")
    
    def stop(self) -> None:
        """Stop all workers in the pool with timeout protection."""
        logger.info(f"Stopping {len(self.workers)} transcription workers...")
        
        # Send sentinel values to stop workers
        try:
            for i in range(self.num_workers):
                self.chunk_queue.put(None)
                logger.debug(f"Sent stop signal to worker {i+1}")
        except Exception as e:
            logger.warning(f"Error sending stop signals: {e}")
        
        # Stop all workers with timeout protection
        stopped_count = 0
        for i, worker in enumerate(self.workers):
            try:
                worker.stop()
                stopped_count += 1
                logger.debug(f"Worker {i+1} stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping worker {i+1}: {e}")
        
        self.workers.clear()
        logger.info(f"Stopped {stopped_count}/{self.num_workers} transcription workers successfully")
    
    def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Add a chunk to the processing queue (legacy method).
        
        Args:
            audio_chunk: Audio data as numpy array
        """
        self.chunk_count += 1
        self.chunk_queue.put((self.chunk_count, audio_chunk))
        logger.debug(f"Added chunk {self.chunk_count} to processing queue")
    
    def add_chunk_with_id(self, chunk_id: str, audio_chunk: np.ndarray, timestamp: float) -> None:
        """
        Add a chunk with specific ID and timestamp to the processing queue.
        
        Args:
            chunk_id: Unique identifier for this chunk
            audio_chunk: Audio data as numpy array
            timestamp: Timestamp when chunk was recorded
        """
        chunk_info = {
            'id': chunk_id,
            'data': audio_chunk,
            'timestamp': timestamp,
            'retry_count': 0
        }
        
        try:
            self.chunk_queue.put(chunk_info, timeout=1.0)
            logger.debug(f"Added chunk {chunk_id} to processing queue")
        except queue.Full:
            logger.warning(f"Chunk queue full, dropping chunk {chunk_id}")
    
    def get_completed_results(self, timeout: float = 0.5) -> list:
        """
        Get all completed transcription results.
        
        Args:
            timeout: Maximum time to wait for results
            
        Returns:
            List of (chunk_id, transcript_text, timestamp) tuples
        """
        results = []
        
        try:
            while True:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
        except queue.Empty:
            pass
        
        return results
    
    def get_streaming_transcript(self) -> str:
        """
        Get the current assembled transcript from completed chunks.
        
        Returns:
            str: Current transcript assembled from streaming results
        """
        with self.streaming_lock:
            if not self.streaming_results:
                return ""
            
            # Sort by timestamp and join
            sorted_results = sorted(self.streaming_results.values(), key=lambda x: x[0])
            transcript_parts = [text for _, text in sorted_results if text.strip()]
            
            return " ".join(transcript_parts)
    
    def _result_callback(self, chunk_id: str, transcribed_text: str, timestamp: float) -> None:
        """
        Callback for worker results.
        
        Args:
            chunk_id: Unique identifier for the chunk
            transcribed_text: Transcribed text from the chunk
            timestamp: Timestamp when chunk was recorded
        """
        # Store in streaming results
        with self.streaming_lock:
            self.streaming_results[chunk_id] = (timestamp, transcribed_text)
        
        # Also put in result queue for immediate retrieval
        try:
            self.result_queue.put((chunk_id, transcribed_text, timestamp), timeout=1.0)
        except queue.Full:
            logger.warning(f"Result queue full, dropping result for chunk {chunk_id}")
        
        # Legacy support: maintain transcribed_segments list
        with self.segments_lock:
            # Convert chunk_id to number if possible for legacy compatibility
            try:
                chunk_number = int(chunk_id.split('_')[-1]) if '_' in chunk_id else int(chunk_id)
                
                # Ensure we have space for this chunk number
                while len(self.transcribed_segments) < chunk_number:
                    self.transcribed_segments.append("")
                
                # Insert the result at the correct position
                self.transcribed_segments[chunk_number - 1] = transcribed_text
            except (ValueError, IndexError):
                # If chunk_id isn't numeric, just append
                self.transcribed_segments.append(transcribed_text)
        
        logger.debug(f"Result for chunk {chunk_id} stored: "
                    f"'{transcribed_text[:50]}...'")
    
    def get_assembled_transcript(self) -> str:
        """
        Get the complete assembled transcript (legacy method).
        
        Returns:
            str: Complete transcript with all processed chunks
        """
        with self.segments_lock:
            # Filter out empty segments and join
            non_empty_segments = [seg for seg in self.transcribed_segments if seg.strip()]
            return " ".join(non_empty_segments)
    
    def get_stats(self) -> dict:
        """
        Get pool statistics.
        
        Returns:
            dict: Combined statistics from all workers
        """
        total_processed = sum(worker.chunks_processed for worker in self.workers)
        total_failed = sum(worker.chunks_failed for worker in self.workers)
        total_time = sum(worker.total_processing_time for worker in self.workers)
        
        return {
            'num_workers': self.num_workers,
            'total_chunks_processed': total_processed,
            'total_chunks_failed': total_failed,
            'total_processing_time': total_time,
            'avg_processing_time': total_time / total_processed if total_processed > 0 else 0.0,
            'success_rate': total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0.0,
            'segments_assembled': len(self.transcribed_segments)
        }
