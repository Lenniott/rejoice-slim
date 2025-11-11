# src/volume_segmenter.py

import threading
import time
from typing import List, Optional, Tuple, Callable
import numpy as np
import logging
from dataclasses import dataclass
from audio_buffer import CircularAudioBuffer, BufferSegmentIterator

logger = logging.getLogger(__name__)

@dataclass
class SegmentInfo:
    """Information about a detected audio segment."""
    start_time: float
    end_time: float
    duration: float
    reason: str  # 'natural_pause', 'max_length', 'forced'
    avg_volume: float
    peak_volume: float
    silence_duration: float

@dataclass
class VolumeConfig:
    """Configuration for volume-based segmentation."""
    min_segment_duration: float = 30.0      # Never break before 30 seconds
    target_segment_duration: float = 45.0   # Prefer breaks around 45 seconds  
    max_segment_duration: float = 90.0      # Always break by 90 seconds
    volume_drop_threshold: float = 0.7      # 30% volume drop required (70% remaining)
    silence_threshold: float = 0.02         # Absolute silence level (RMS)
    min_pause_duration: float = 0.5         # Minimum pause length to consider
    analysis_window: float = 1.0            # 1-second RMS calculation windows
    lookback_window: float = 5.0            # Seconds to look back for volume baseline

class VolumeSegmenter:
    """
    Intelligent audio segmentation using volume analysis.
    
    Analyzes audio in real-time to find natural speaking boundaries
    based on volume drops and silence detection, creating segments
    of optimal length for Whisper transcription.
    """
    
    def __init__(self, 
                 audio_buffer: CircularAudioBuffer,
                 config: Optional[VolumeConfig] = None,
                 verbose: bool = False):
        """
        Initialize the volume segmenter.
        
        Args:
            audio_buffer: CircularAudioBuffer to analyze
            config: VolumeConfig for segmentation parameters
            verbose: Enable verbose logging
        """
        self.audio_buffer = audio_buffer
        self.config = config or VolumeConfig()
        self.verbose = verbose
        
        # State tracking
        self.lock = threading.RLock()
        self.current_segment_start = 0.0
        self.last_analysis_time = 0.0
        self.volume_history = []  # List of (timestamp, rms_volume) tuples
        self.detected_segments = []  # List of SegmentInfo objects
        self.is_analyzing = False
        
        # Callbacks
        self.segment_ready_callback: Optional[Callable[[SegmentInfo], None]] = None
        
        logger.info(f"VolumeSegmenter initialized with config: "
                   f"min={self.config.min_segment_duration}s, "
                   f"target={self.config.target_segment_duration}s, "
                   f"max={self.config.max_segment_duration}s")
    
    def start_analysis(self) -> None:
        """Start volume analysis for segmentation."""
        with self.lock:
            self.current_segment_start = 0.0
            self.last_analysis_time = 0.0
            self.volume_history = []
            self.detected_segments = []
            self.is_analyzing = True
            
        logger.info("VolumeSegmenter: Started analysis")
    
    def stop_analysis(self) -> None:
        """Stop volume analysis."""
        with self.lock:
            self.is_analyzing = False
            
        logger.info("VolumeSegmenter: Stopped analysis")
    
    def analyze_and_segment(self) -> List[SegmentInfo]:
        """
        Analyze current audio and detect any new segments.
        
        This should be called periodically during recording to
        detect segments as they become available.
        
        Returns:
            List of newly detected segments
        """
        if not self.is_analyzing:
            return []
        
        with self.lock:
            new_segments = []
            current_time = self.audio_buffer.get_recording_duration()
            
            # Skip if no new audio data
            if current_time <= self.last_analysis_time:
                return []
            
            # Update volume history with new analysis windows
            new_segments.extend(self._update_volume_history(current_time))
            
            # Check for segment boundaries
            segment = self._check_for_segment_boundary(current_time)
            if segment:
                new_segments.append(segment)
                self._finalize_segment(segment)
            
            self.last_analysis_time = current_time
            return new_segments
    
    def flush_remaining_segment(self) -> Optional[SegmentInfo]:
        """
        Force completion of the current segment.
        
        Called when recording stops to ensure the final
        segment is captured.
        
        Returns:
            Final segment if any audio remains
        """
        with self.lock:
            if not self.is_analyzing:
                return None
                
            current_time = self.audio_buffer.get_recording_duration()
            remaining_duration = current_time - self.current_segment_start
            
            if remaining_duration < 5.0:  # Skip very short final segments
                logger.info(f"VolumeSegmenter: Skipping short final segment ({remaining_duration:.1f}s)")
                return None
            
            # Create final segment
            segment = SegmentInfo(
                start_time=self.current_segment_start,
                end_time=current_time,
                duration=remaining_duration,
                reason='final_flush',
                avg_volume=self._get_average_volume_in_range(self.current_segment_start, current_time),
                peak_volume=self._get_peak_volume_in_range(self.current_segment_start, current_time),
                silence_duration=0.0
            )
            
            self._finalize_segment(segment)
            logger.info(f"VolumeSegmenter: Flushed final segment: {segment.duration:.1f}s")
            return segment
    
    def _update_volume_history(self, current_time: float) -> List[SegmentInfo]:
        """Update volume history with new analysis windows."""
        analysis_start = max(self.last_analysis_time, 0.0)
        
        # Create iterator for new audio segments
        iterator = BufferSegmentIterator(
            self.audio_buffer,
            window_size=self.config.analysis_window,
            overlap=0.0
        )
        iterator.reset_to_position(analysis_start)
        
        new_segments = []
        
        try:
            while iterator.has_next():
                timestamp, audio_segment = next(iterator)
                
                # Skip if we've moved past current time
                if timestamp >= current_time:
                    break
                
                # Calculate RMS volume
                rms_volume = self._calculate_rms(audio_segment)
                self.volume_history.append((timestamp, rms_volume))
                
                if self.verbose:
                    logger.debug(f"Volume analysis: t={timestamp:.1f}s, RMS={rms_volume:.6f}")
                
        except StopIteration:
            pass
        
        # Trim old volume history to save memory
        cutoff_time = current_time - (self.config.lookback_window * 2)
        self.volume_history = [(t, v) for t, v in self.volume_history if t >= cutoff_time]
        
        return new_segments
    
    def _check_for_segment_boundary(self, current_time: float) -> Optional[SegmentInfo]:
        """Check if we should create a segment boundary."""
        current_duration = current_time - self.current_segment_start
        
        # Debug logging
        if self.verbose and current_duration > 10 and current_duration % 5 < 0.5:  # Log every 5s after 10s
            logger.info(f"VolumeSegmenter: {current_duration:.1f}s into segment (max={self.config.max_segment_duration}s)")
        
        # Force break if we've hit maximum duration
        if current_duration >= self.config.max_segment_duration:
            if self.verbose:
                logger.info(f"VolumeSegmenter: TRIGGERING max_length segment at {current_duration:.1f}s")
            return self._create_segment(current_time, 'max_length')
        
        # Don't check for natural breaks until minimum duration
        if current_duration < self.config.min_segment_duration:
            return None
        
        # Look for natural pause after target duration
        if current_duration >= self.config.target_segment_duration:
            pause_info = self._detect_natural_pause(current_time)
            if pause_info:
                pause_end_time, silence_duration = pause_info
                if self.verbose:
                    logger.info(f"VolumeSegmenter: Found natural pause at {pause_end_time:.1f}s")
                return self._create_segment(pause_end_time, 'natural_pause', silence_duration)
        
        return None
    
    def _detect_natural_pause(self, current_time: float) -> Optional[Tuple[float, float]]:
        """
        Detect natural speaking pauses using volume analysis.
        
        Returns:
            Tuple of (pause_end_time, silence_duration) if found
        """
        if len(self.volume_history) < 3:
            return None
        
        # Look for volume drops in recent history
        lookback_start = current_time - self.config.lookback_window
        recent_volumes = [(t, v) for t, v in self.volume_history if t >= lookback_start]
        
        if len(recent_volumes) < 3:
            return None
        
        # Calculate baseline volume from earlier in the segment
        baseline_start = max(self.current_segment_start, current_time - self.config.lookback_window)
        baseline_volumes = [v for t, v in recent_volumes if baseline_start <= t <= current_time - 2.0]
        
        if not baseline_volumes:
            return None
            
        baseline_volume = np.mean(baseline_volumes)
        
        # Look for consecutive low-volume windows
        silence_start = None
        silence_end = None
        
        for i, (timestamp, volume) in enumerate(recent_volumes):
            is_silence = (volume < self.config.silence_threshold or 
                         volume < baseline_volume * self.config.volume_drop_threshold)
            
            if is_silence and silence_start is None:
                silence_start = timestamp
            elif not is_silence and silence_start is not None:
                silence_end = timestamp
                break
        
        # If we're still in silence, use current time as end
        if silence_start is not None and silence_end is None:
            silence_end = current_time
        
        # Check if we found a valid pause
        if silence_start is not None and silence_end is not None:
            silence_duration = silence_end - silence_start
            
            if silence_duration >= self.config.min_pause_duration:
                if self.verbose:
                    logger.debug(f"Natural pause detected: {silence_start:.1f}s-{silence_end:.1f}s "
                               f"(duration: {silence_duration:.1f}s, baseline: {baseline_volume:.6f})")
                return silence_end, silence_duration
        
        return None
    
    def _create_segment(self, end_time: float, reason: str, silence_duration: float = 0.0) -> SegmentInfo:
        """Create a SegmentInfo object for the current segment."""
        duration = end_time - self.current_segment_start
        
        segment = SegmentInfo(
            start_time=self.current_segment_start,
            end_time=end_time,
            duration=duration,
            reason=reason,
            avg_volume=self._get_average_volume_in_range(self.current_segment_start, end_time),
            peak_volume=self._get_peak_volume_in_range(self.current_segment_start, end_time),
            silence_duration=silence_duration
        )
        
        if self.verbose:
            logger.info(f"Segment created: {segment.start_time:.1f}s-{segment.end_time:.1f}s "
                       f"({segment.duration:.1f}s, {segment.reason})")
        
        return segment
    
    def _finalize_segment(self, segment: SegmentInfo) -> None:
        """Finalize a segment and prepare for the next one."""
        self.detected_segments.append(segment)
        self.current_segment_start = segment.end_time
        
        # Trigger callback if set
        if self.segment_ready_callback:
            try:
                self.segment_ready_callback(segment)
            except Exception as e:
                logger.error(f"Error in segment callback: {e}")
    
    def _get_average_volume_in_range(self, start_time: float, end_time: float) -> float:
        """Calculate average RMS volume in time range."""
        relevant_volumes = [v for t, v in self.volume_history 
                           if start_time <= t <= end_time]
        return np.mean(relevant_volumes) if relevant_volumes else 0.0
    
    def _get_peak_volume_in_range(self, start_time: float, end_time: float) -> float:
        """Calculate peak RMS volume in time range."""
        relevant_volumes = [v for t, v in self.volume_history 
                           if start_time <= t <= end_time]
        return np.max(relevant_volumes) if relevant_volumes else 0.0
    
    def _calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) volume of audio data."""
        if len(audio_data) == 0:
            return 0.0
        return np.sqrt(np.mean(audio_data ** 2))
    
    def set_segment_callback(self, callback: Callable[[SegmentInfo], None]) -> None:
        """Set callback to be called when a new segment is ready."""
        self.segment_ready_callback = callback
        logger.info("VolumeSegmenter: Segment callback set")
    
    def get_stats(self) -> dict:
        """Get segmentation statistics."""
        with self.lock:
            current_segment_duration = 0.0
            if self.is_analyzing and self.audio_buffer.get_recording_duration() > 0:
                current_segment_duration = (self.audio_buffer.get_recording_duration() - 
                                          self.current_segment_start)
            
            segment_durations = [s.duration for s in self.detected_segments]
            
            return {
                'is_analyzing': self.is_analyzing,
                'segments_detected': len(self.detected_segments),
                'current_segment_duration': current_segment_duration,
                'avg_segment_duration': np.mean(segment_durations) if segment_durations else 0.0,
                'min_segment_duration': np.min(segment_durations) if segment_durations else 0.0,
                'max_segment_duration': np.max(segment_durations) if segment_durations else 0.0,
                'natural_pauses': len([s for s in self.detected_segments if s.reason == 'natural_pause']),
                'forced_breaks': len([s for s in self.detected_segments if s.reason == 'max_length']),
                'volume_history_length': len(self.volume_history),
                'config': {
                    'min_duration': self.config.min_segment_duration,
                    'target_duration': self.config.target_segment_duration,
                    'max_duration': self.config.max_segment_duration,
                    'volume_drop_threshold': self.config.volume_drop_threshold,
                    'silence_threshold': self.config.silence_threshold
                }
            }
    
    def get_detected_segments(self) -> List[SegmentInfo]:
        """Get all detected segments."""
        with self.lock:
            return self.detected_segments.copy()
    
    def clear_segments(self) -> None:
        """Clear detected segments (useful for testing)."""
        with self.lock:
            self.detected_segments = []
            logger.info("VolumeSegmenter: Cleared detected segments")

class SegmentProcessor:
    """
    Processes segments for transcription with Whisper.
    
    Extracts audio data for segments and queues them for
    transcription processing.
    """
    
    def __init__(self, audio_buffer: CircularAudioBuffer):
        """Initialize segment processor."""
        self.audio_buffer = audio_buffer
        self.processed_segments = {}  # segment_id -> audio_data
        self.lock = threading.RLock()
        
    def extract_segment_audio(self, segment: SegmentInfo) -> Optional[np.ndarray]:
        """
        Extract audio data for a segment.
        
        Args:
            segment: SegmentInfo with timing information
            
        Returns:
            Audio data as numpy array, or None if not available
        """
        with self.lock:
            # Add retry logic to handle race condition between 
            # segment creation and audio buffer writes
            max_retries = 10  # Increased retries
            retry_delay = 0.2  # 200ms delay
            
            for attempt in range(max_retries):
                audio_data = self.audio_buffer.read_segment(
                    segment.start_time, 
                    segment.duration
                )
                
                if audio_data is not None:
                    segment_id = f"{segment.start_time:.1f}-{segment.end_time:.1f}"
                    self.processed_segments[segment_id] = audio_data
                    
                    logger.debug(f"Extracted audio for segment {segment_id}: "
                               f"{len(audio_data)} samples (attempt {attempt + 1})")
                    return audio_data
                    
                elif attempt < max_retries - 1:
                    # Brief wait for audio buffer to catch up
                    logger.debug(f"Audio not ready for segment {segment.start_time:.1f}s, "
                               f"retrying ({attempt + 1}/{max_retries}) in {retry_delay}s...")
                    import time
                    time.sleep(retry_delay)
            
            logger.warning(f"Failed to extract audio for segment {segment.start_time:.1f}s-{segment.end_time:.1f}s "
                          f"after {max_retries} attempts (waited {max_retries * retry_delay:.1f}s)")
            return None
    
    def get_segment_stats(self) -> dict:
        """Get processor statistics."""
        with self.lock:
            return {
                'processed_segments': len(self.processed_segments),
                'total_samples': sum(len(data) for data in self.processed_segments.values())
            }