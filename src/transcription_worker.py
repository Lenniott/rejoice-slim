# src/transcription_worker.py

import time
import threading
import queue
import logging
from typing import List, Optional, Callable
import numpy as np
import whisper

logger = logging.getLogger(__name__)

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
              result_callback: Callable[[str, int], None]) -> None:
        """
        Start the worker thread.
        
        Args:
            chunk_queue: Queue containing audio chunks to process
            result_callback: Function to call with transcription results
                            Signature: callback(transcribed_text, chunk_number)
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
        """Stop the worker thread gracefully."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            logger.info(f"TranscriptionWorker {self.worker_id} stopped")
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes chunks from the queue."""
        logger.debug(f"Worker {self.worker_id} entering main loop")
        
        while not self.stop_event.is_set():
            try:
                # Get chunk with timeout to allow checking stop_event
                chunk_data = self.chunk_queue.get(timeout=1.0)
                
                if chunk_data is None:  # Sentinel value to stop
                    break
                
                chunk_number, audio_chunk = chunk_data
                self._process_chunk(audio_chunk, chunk_number)
                
            except queue.Empty:
                # Timeout - check if we should stop
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} unexpected error: {e}")
                break
        
        logger.debug(f"Worker {self.worker_id} exiting main loop")
    
    def _process_chunk(self, audio_chunk: np.ndarray, chunk_number: int) -> None:
        """
        Process a single audio chunk with retry logic.
        
        Args:
            audio_chunk: Audio data as numpy array
            chunk_number: Sequential chunk number for logging
        """
        start_time = time.time()
        
        for attempt in range(self.max_retry_attempts + 1):
            try:
                # Transcribe the chunk
                if self.whisper_language and self.whisper_language.lower() != "auto":
                    result = self.whisper_model.transcribe(
                        audio_chunk, 
                        fp16=False, 
                        language=self.whisper_language
                    )
                else:
                    result = self.whisper_model.transcribe(audio_chunk, fp16=False)
                
                transcribed_text = result["text"].strip()
                
                # Success - send result
                if transcribed_text:
                    self.result_callback(transcribed_text, chunk_number)
                    logger.debug(f"Worker {self.worker_id} chunk {chunk_number} "
                               f"transcribed successfully: '{transcribed_text[:50]}...'")
                else:
                    # Empty transcription - still count as success
                    self.result_callback("", chunk_number)
                    logger.debug(f"Worker {self.worker_id} chunk {chunk_number} "
                               f"produced empty transcription")
                
                # Update statistics
                self.chunks_processed += 1
                self.total_processing_time += time.time() - start_time
                return
                
            except Exception as e:
                if attempt < self.max_retry_attempts:
                    # Retry with exponential backoff
                    delay = self.retry_delay_base * (2 ** attempt)
                    logger.warning(f"Worker {self.worker_id} chunk {chunk_number} "
                                 f"attempt {attempt + 1} failed: {e}. "
                                 f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # All retries exhausted
                    error_placeholder = f"[--- Transcription failed for this segment (chunk {chunk_number}) ---]"
                    self.result_callback(error_placeholder, chunk_number)
                    self.chunks_failed += 1
                    
                    logger.error(f"Worker {self.worker_id} chunk {chunk_number} "
                               f"failed after {self.max_retry_attempts} retries: {e}")
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
        
        # Result collection
        self.transcribed_segments = []
        self.segments_lock = threading.Lock()
        self.chunk_count = 0
        
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
        """Stop all workers in the pool."""
        # Send sentinel values to stop workers
        for _ in range(self.num_workers):
            self.chunk_queue.put(None)
        
        # Stop all workers
        for worker in self.workers:
            worker.stop()
        
        self.workers.clear()
        logger.info("All transcription workers stopped")
    
    def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Add a chunk to the processing queue.
        
        Args:
            audio_chunk: Audio data as numpy array
        """
        self.chunk_count += 1
        self.chunk_queue.put((self.chunk_count, audio_chunk))
        logger.debug(f"Added chunk {self.chunk_count} to processing queue")
    
    def _result_callback(self, transcribed_text: str, chunk_number: int) -> None:
        """
        Callback for worker results.
        
        Args:
            transcribed_text: Transcribed text from the chunk
            chunk_number: Sequential chunk number
        """
        with self.segments_lock:
            # Ensure we have space for this chunk number
            while len(self.transcribed_segments) < chunk_number:
                self.transcribed_segments.append("")
            
            # Insert the result at the correct position
            self.transcribed_segments[chunk_number - 1] = transcribed_text
            
            logger.debug(f"Result for chunk {chunk_number} stored: "
                        f"'{transcribed_text[:50]}...'")
    
    def get_assembled_transcript(self) -> str:
        """
        Get the complete assembled transcript.
        
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
