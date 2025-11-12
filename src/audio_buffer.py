# src/audio_buffer.py

import threading
import time
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CircularAudioBuffer:
    """
    A thread-safe circular buffer for real-time audio processing.
    
    Provides a fixed-size rolling buffer that efficiently stores audio data
    for volume-based segmentation analysis while maintaining memory efficiency.
    """
    
    def __init__(self, 
                 capacity_seconds: int = 300,  # 5 minutes default
                 sample_rate: int = 16000,
                 channels: int = 1,
                 dtype: str = 'float32'):
        """
        Initialize the circular audio buffer.
        
        Args:
            capacity_seconds: Maximum buffer capacity in seconds
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            dtype: Audio data type ('float32', 'int16', etc.)
        """
        self.capacity_seconds = capacity_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = getattr(np, dtype)
        
        # Calculate buffer size in samples
        self.buffer_size = capacity_seconds * sample_rate * channels
        
        # Initialize circular buffer
        self.buffer = np.zeros(self.buffer_size, dtype=self.dtype)
        
        # Threading and state management
        self.lock = threading.RLock()
        self.write_position = 0
        self.total_samples_written = 0
        self.start_time = None
        self.is_recording = False
        
        logger.info(f"CircularAudioBuffer initialized: {capacity_seconds}s capacity, "
                   f"{sample_rate}Hz, {channels} channel(s), {dtype}")
    
    def start_recording(self) -> None:
        """Start a new recording session."""
        with self.lock:
            self.write_position = 0
            self.total_samples_written = 0
            self.start_time = time.time()
            self.is_recording = True
            
        logger.info("CircularAudioBuffer: Recording session started")
    
    def stop_recording(self) -> None:
        """Stop the current recording session."""
        with self.lock:
            self.is_recording = False
            
        logger.info(f"CircularAudioBuffer: Recording session stopped after "
                   f"{self.get_recording_duration():.1f} seconds")
    
    def write(self, audio_data: np.ndarray) -> None:
        """
        Write audio data to the circular buffer.
        
        Args:
            audio_data: Audio samples to write (1D numpy array)
        """
        if not self.is_recording:
            return
            
        with self.lock:
            data_length = len(audio_data)
            
            # Handle buffer wrap-around
            end_position = self.write_position + data_length
            
            if end_position <= self.buffer_size:
                # Simple case: data fits without wrapping
                self.buffer[self.write_position:end_position] = audio_data
            else:
                # Wrap-around case: split data across buffer boundary
                first_part_size = self.buffer_size - self.write_position
                second_part_size = data_length - first_part_size
                
                self.buffer[self.write_position:] = audio_data[:first_part_size]
                self.buffer[:second_part_size] = audio_data[first_part_size:]
            
            # Update position and tracking
            self.write_position = end_position % self.buffer_size
            self.total_samples_written += data_length
            
            logger.debug(f"CircularAudioBuffer: Wrote {data_length} samples, "
                        f"total: {self.total_samples_written}, "
                        f"position: {self.write_position}")
    
    def read_segment(self, start_time_offset: float, duration: float) -> Optional[np.ndarray]:
        """
        Read a segment of audio data by time offset.
        
        Args:
            start_time_offset: Start time in seconds from recording start
            duration: Duration in seconds to read
            
        Returns:
            Audio data segment or None if not available
        """
        with self.lock:
            if not self.start_time:
                return None
                
            # Convert time to sample positions
            start_sample = int(start_time_offset * self.sample_rate * self.channels)
            duration_samples = int(duration * self.sample_rate * self.channels)
            end_sample = start_sample + duration_samples
            
            # Check if requested data is still in buffer
            if start_sample < self.total_samples_written - self.buffer_size:
                logger.warning(f"Requested data too old: start={start_sample}, "
                             f"buffer_start={self.total_samples_written - self.buffer_size}")
                return None
                
            if end_sample > self.total_samples_written:
                logger.warning(f"Requested data not yet available: end={end_sample}, "
                             f"written={self.total_samples_written}")
                return None
            
            # Calculate buffer positions
            buffer_start = start_sample % self.buffer_size
            buffer_end = (start_sample + duration_samples) % self.buffer_size
            
            # Extract data (handle wrap-around)
            if buffer_start < buffer_end:
                # Simple case: no wrap-around
                segment = self.buffer[buffer_start:buffer_end].copy()
            else:
                # Wrap-around case: concatenate two parts
                first_part = self.buffer[buffer_start:].copy()
                second_part = self.buffer[:buffer_end].copy()
                segment = np.concatenate([first_part, second_part])
            
            logger.debug(f"CircularAudioBuffer: Read segment {start_time_offset:.1f}s "
                        f"+ {duration:.1f}s ({len(segment)} samples)")
            
            return segment
    
    def get_latest_segment(self, duration: float) -> Optional[np.ndarray]:
        """
        Get the most recent audio data of specified duration.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Latest audio segment or None if not enough data
        """
        with self.lock:
            if not self.is_recording or not self.start_time:
                return None
                
            # Calculate current audio data duration (not wall clock time)
            audio_data_duration = self.total_samples_written / (self.sample_rate * self.channels)
            
            if audio_data_duration < duration:
                return None
                
            start_offset = audio_data_duration - duration
            return self.read_segment(start_offset, duration)
    
    def get_recording_duration(self) -> float:
        """
        Get the current recording duration in seconds based on captured samples.
        """
        with self.lock:
            if self.total_samples_written == 0:
                if self.is_recording and self.start_time:
                    # No audio written yet, fall back to wall time
                    return time.time() - self.start_time
                return 0.0
            
            return self.total_samples_written / (self.sample_rate * self.channels)
    
    def get_available_duration(self) -> float:
        """
        Get the duration of data currently available in buffer.
        
        Returns:
            Available duration in seconds
        """
        with self.lock:
            available_samples = min(self.total_samples_written, self.buffer_size)
            return available_samples / (self.sample_rate * self.channels)
    
    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        buffer_bytes = self.buffer.nbytes
        return buffer_bytes / (1024 * 1024)
    
    def get_stats(self) -> dict:
        """
        Get comprehensive buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        with self.lock:
            return {
                'is_recording': self.is_recording,
                'capacity_seconds': self.capacity_seconds,
                'recording_duration': self.get_recording_duration(),
                'available_duration': self.get_available_duration(),
                'total_samples_written': self.total_samples_written,
                'write_position': self.write_position,
                'buffer_size': self.buffer_size,
                'memory_usage_mb': self.get_memory_usage_mb(),
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'buffer_utilization': min(self.total_samples_written / self.buffer_size, 1.0)
            }
    
    def clear(self) -> None:
        """Clear the buffer and reset all state."""
        with self.lock:
            self.buffer.fill(0)
            self.write_position = 0
            self.total_samples_written = 0
            self.start_time = None
            self.is_recording = False
            
        logger.info("CircularAudioBuffer: Buffer cleared and reset")

class BufferSegmentIterator:
    """
    Iterator for processing buffer segments in chronological order.
    
    Useful for volume analysis and segmentation algorithms that need
    to process audio data in overlapping or non-overlapping windows.
    """
    
    def __init__(self, 
                 buffer: CircularAudioBuffer,
                 window_size: float = 1.0,  # 1 second windows
                 overlap: float = 0.0):     # No overlap by default
        """
        Initialize the segment iterator.
        
        Args:
            buffer: CircularAudioBuffer instance
            window_size: Size of each window in seconds
            overlap: Overlap between windows in seconds
        """
        self.buffer = buffer
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        self.current_position = 0.0
    
    def __iter__(self):
        """Reset iterator to beginning."""
        self.current_position = 0.0
        return self
    
    def __next__(self) -> Tuple[float, np.ndarray]:
        """
        Get the next audio segment.
        
        Returns:
            Tuple of (timestamp, audio_data)
            
        Raises:
            StopIteration when no more segments available
        """
        available_duration = self.buffer.get_available_duration()
        
        if self.current_position + self.window_size > available_duration:
            raise StopIteration
        
        segment = self.buffer.read_segment(self.current_position, self.window_size)
        if segment is None:
            raise StopIteration
        
        timestamp = self.current_position
        self.current_position += self.step_size
        
        return timestamp, segment
    
    def reset_to_position(self, position: float) -> None:
        """Reset iterator to a specific time position."""
        self.current_position = max(0.0, position)
    
    def has_next(self) -> bool:
        """Check if more segments are available."""
        available_duration = self.buffer.get_available_duration()
        return self.current_position + self.window_size <= available_duration