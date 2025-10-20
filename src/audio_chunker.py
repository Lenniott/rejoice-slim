# src/audio_chunker.py

import numpy as np
import threading
import time
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)

class AudioChunker:
    """
    Manages overlapping audio chunk creation for real-time processing.
    
    Implements a rolling buffer strategy to ensure no audio data is lost
    at chunk boundaries. Chunks are created with configurable duration
    and overlap to prevent word splitting.
    """
    
    def __init__(self, 
                 chunk_duration_seconds: float = 10.0,
                 overlap_seconds: float = 1.0,
                 sample_rate: int = 16000):
        """
        Initialize the audio chunker.
        
        Args:
            chunk_duration_seconds: Duration of each chunk in seconds
            overlap_seconds: Overlap between consecutive chunks in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.chunk_duration_seconds = chunk_duration_seconds
        self.overlap_seconds = overlap_seconds
        self.sample_rate = sample_rate
        
        # Calculate samples for duration and overlap
        self.chunk_samples = int(chunk_duration_seconds * sample_rate)
        self.overlap_samples = int(overlap_seconds * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples
        
        # Rolling buffer to maintain overlap
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # Track timing for logging
        self.chunk_count = 0
        self.start_time = None
        
        logger.info(f"AudioChunker initialized: {chunk_duration_seconds}s chunks, "
                   f"{overlap_seconds}s overlap, {sample_rate}Hz")
    
    def add_audio_data(self, audio_data: np.ndarray) -> None:
        """
        Add new audio data to the rolling buffer.
        
        Args:
            audio_data: New audio samples as numpy array
        """
        with self.buffer_lock:
            self.buffer = np.concatenate([self.buffer, audio_data])
            
            if self.start_time is None:
                self.start_time = time.time()
    
    def get_ready_chunks(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields complete chunks as they become available.
        
        Yields:
            np.ndarray: Complete audio chunk ready for processing
        """
        while True:
            with self.buffer_lock:
                if len(self.buffer) >= self.chunk_samples:
                    # Extract chunk
                    chunk = self.buffer[:self.chunk_samples].copy()
                    
                    # Remove step_samples from buffer (keeping overlap)
                    self.buffer = self.buffer[self.step_samples:]
                    
                    # Update chunk metadata
                    self.chunk_count += 1
                    chunk_start_time = (self.chunk_count - 1) * self.step_samples / self.sample_rate
                    
                    logger.debug(f"Chunk {self.chunk_count} ready: "
                               f"{len(chunk)} samples, start at {chunk_start_time:.2f}s")
                    
                    yield chunk
                else:
                    # Not enough data for a complete chunk
                    break
    
    def get_final_chunk(self) -> Optional[np.ndarray]:
        """
        Get the final partial chunk when recording stops.
        
        Returns:
            np.ndarray or None: Final chunk if there's remaining audio data
        """
        with self.buffer_lock:
            if len(self.buffer) > 0:
                # Return whatever is left in the buffer
                final_chunk = self.buffer.copy()
                self.buffer = np.array([], dtype=np.float32)
                
                logger.info(f"Final chunk: {len(final_chunk)} samples "
                          f"({len(final_chunk) / self.sample_rate:.2f}s)")
                return final_chunk
            
            return None
    
    def reset(self) -> None:
        """Reset the chunker state for a new recording session."""
        with self.buffer_lock:
            self.buffer = np.array([], dtype=np.float32)
            self.chunk_count = 0
            self.start_time = None
        
        logger.info("AudioChunker reset for new recording session")
    
    def get_stats(self) -> dict:
        """
        Get current chunker statistics.
        
        Returns:
            dict: Statistics including chunk count, buffer size, etc.
        """
        with self.buffer_lock:
            return {
                'chunk_count': self.chunk_count,
                'buffer_samples': len(self.buffer),
                'buffer_duration_seconds': len(self.buffer) / self.sample_rate,
                'chunk_duration_seconds': self.chunk_duration_seconds,
                'overlap_seconds': self.overlap_seconds,
                'sample_rate': self.sample_rate
            }
