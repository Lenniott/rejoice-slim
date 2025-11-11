# src/streaming_recorder.py

import threading
import time
import wave
import os
from typing import Optional, Callable
import numpy as np
import sounddevice as sd
import logging
from dataclasses import dataclass

from audio_buffer import CircularAudioBuffer
from volume_segmenter import VolumeSegmenter, VolumeConfig, SegmentInfo
from vad_service import VADService

logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for streaming recording."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    device: Optional[int] = None

class StreamingRecorder:
    """
    Dual-output audio recorder with real-time segmentation.
    
    Records audio to both a master file and a circular buffer,
    with integrated volume analysis and VAD for auto-stop.
    """
    
    def __init__(self, 
                 config: Optional[StreamingConfig] = None,
                 volume_config: Optional[VolumeConfig] = None,
                 auto_stop_enabled: bool = True,
                 auto_stop_duration: int = 120):
        """
        Initialize the streaming recorder.
        
        Args:
            config: StreamingConfig for audio parameters
            volume_config: VolumeConfig for segmentation
            auto_stop_enabled: Enable VAD auto-stop
            auto_stop_duration: Auto-stop silence duration in seconds
        """
        self.config = config or StreamingConfig()
        self.volume_config = volume_config or VolumeConfig()
        self.auto_stop_enabled = auto_stop_enabled
        self.auto_stop_duration = auto_stop_duration
        
        # Audio recording components
        self.audio_stream = None
        self.master_wav_file = None
        self.master_wave_file = None
        
        # Streaming components
        self.audio_buffer = CircularAudioBuffer(
            capacity_seconds=self.config.buffer_capacity_seconds,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            dtype='float32'
        )
        
        self.volume_segmenter = VolumeSegmenter(
            audio_buffer=self.audio_buffer,
            config=self.volume_config,
            verbose=self.config.verbose
        )
        
        # VAD for auto-stop
        self.vad_service = None
        if self.auto_stop_enabled:
            # Convert to chunk duration for VAD (assumes 10s chunks from original system)
            chunk_duration = 10.0  
            silence_chunks = int(self.auto_stop_duration / chunk_duration)
            self.vad_service = VADService(
                silence_threshold_chunks=silence_chunks,
                chunk_duration_seconds=chunk_duration
            )
        
        # State management
        self.lock = threading.RLock()
        self.is_recording = False
        self.recording_thread = None
        self.start_time = None
        self.total_frames_recorded = 0
        
        # Callbacks
        self.segment_callback: Optional[Callable[[SegmentInfo, np.ndarray], None]] = None
        self.auto_stop_callback: Optional[Callable[[], None]] = None
        
        logger.info(f"StreamingRecorder initialized: {self.config.sample_rate}Hz, "
                   f"{self.config.channels} channel(s), auto-stop: {auto_stop_enabled}")
    
    def start_recording(self, output_file_path: str) -> bool:
        """
        Start recording to both master file and buffer.
        
        Args:
            output_file_path: Path for master audio file
            
        Returns:
            True if recording started successfully
        """
        with self.lock:
            if self.is_recording:
                logger.warning("Already recording")
                return False
            
            try:
                # Initialize audio interface
                self.audio_interface = pyaudio.PyAudio()
                
                # Open master file for writing
                self.master_file_path = output_file_path
                self.master_wave_file = wave.open(output_file_path, 'wb')
                self.master_wave_file.setnchannels(self.config.channels)
                self.master_wave_file.setsampwidth(2)  # 16-bit
                self.master_wave_file.setframerate(self.config.sample_rate)
                
                # Start audio stream
                self.audio_stream = self.audio_interface.open(
                    format=self.config.audio_format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.config.chunk_size,
                    stream_callback=None  # We'll read manually for more control
                )
                
                # Start components
                self.audio_buffer.start_recording()
                self.volume_segmenter.start_analysis()
                
                if self.vad_service:
                    self.vad_service.start_recording()
                    self.vad_service.set_auto_stop_callback(self._on_auto_stop_triggered)
                
                # Start recording thread
                self.is_recording = True
                self.start_time = time.time()
                self.total_frames_recorded = 0
                self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
                self.recording_thread.start()
                
                logger.info(f"StreamingRecorder: Recording started to {output_file_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                self._cleanup_resources()
                return False
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and finalize files.
        
        Returns:
            Path to master audio file if successful
        """
        with self.lock:
            if not self.is_recording:
                logger.warning("Not currently recording")
                return None
            
            logger.info("StreamingRecorder: Stopping recording...")
            self.is_recording = False
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=5.0)
                
                if self.recording_thread.is_alive():
                    logger.warning("Recording thread didn't stop cleanly")
            
            # Stop components
            self.audio_buffer.stop_recording()
            self.volume_segmenter.stop_analysis()
            
            if self.vad_service:
                self.vad_service.stop_recording()
            
            # Process final segment
            final_segment = self.volume_segmenter.flush_remaining_segment()
            if final_segment and self.segment_callback:
                audio_data = self.audio_buffer.read_segment(
                    final_segment.start_time,
                    final_segment.duration
                )
                if audio_data is not None:
                    self.segment_callback(final_segment, audio_data)
            
            # Cleanup resources
            master_file_path = self.master_file_path
            self._cleanup_resources()
            
            duration = time.time() - self.start_time if self.start_time else 0
            logger.info(f"StreamingRecorder: Recording stopped after {duration:.1f} seconds")
            
            return master_file_path
    
    def _recording_loop(self) -> None:
        """Main recording loop running in separate thread."""
        logger.debug("StreamingRecorder: Recording loop started")
        
        try:
            while self.is_recording and self.audio_stream:
                # Read audio chunk
                try:
                    audio_data = self.audio_stream.read(
                        self.config.chunk_size, 
                        exception_on_overflow=False
                    )
                    
                    if not audio_data:
                        continue
                        
                    # Convert to numpy array
                    audio_array_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array_float32 = audio_array_int16.astype(np.float32) / 32768.0
                    
                    # Write to master file
                    if self.master_wave_file:
                        self.master_wave_file.writeframes(audio_data)
                        self.total_frames_recorded += len(audio_array_int16)
                    
                    # Write to circular buffer
                    self.audio_buffer.write(audio_array_float32)
                    
                    # VAD analysis for auto-stop
                    if self.vad_service:
                        self.vad_service.analyze_chunk(audio_array_float32)
                    
                    # Volume segmentation analysis
                    new_segments = self.volume_segmenter.analyze_and_segment()
                    for segment in new_segments:
                        self._process_new_segment(segment)
                    
                    # Optional progress feedback
                    if self.config.verbose:
                        self._log_progress()
                    
                except Exception as e:
                    if self.is_recording:  # Only log if we're supposed to be recording
                        logger.error(f"Error in recording loop: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Fatal error in recording loop: {e}")
        finally:
            logger.debug("StreamingRecorder: Recording loop ended")
    
    def _process_new_segment(self, segment: SegmentInfo) -> None:
        """Process a newly detected segment."""
        try:
            # Extract audio data for the segment
            audio_data = self.audio_buffer.read_segment(
                segment.start_time,
                segment.duration
            )
            
            if audio_data is not None and self.segment_callback:
                if self.config.verbose:
                    logger.info(f"New segment ready: {segment.start_time:.1f}s-{segment.end_time:.1f}s "
                              f"({segment.duration:.1f}s, {segment.reason})")
                
                self.segment_callback(segment, audio_data)
            
        except Exception as e:
            logger.error(f"Error processing segment: {e}")
    
    def _on_auto_stop_triggered(self) -> None:
        """Called when VAD detects prolonged silence."""
        logger.info("StreamingRecorder: Auto-stop triggered by silence detection")
        
        if self.auto_stop_callback:
            try:
                self.auto_stop_callback()
            except Exception as e:
                logger.error(f"Error in auto-stop callback: {e}")
        
        # Stop recording
        threading.Thread(target=self.stop_recording, daemon=True).start()
    
    def _log_progress(self) -> None:
        """Log progress information in verbose mode."""
        if not self.start_time:
            return
            
        current_duration = time.time() - self.start_time
        buffer_stats = self.audio_buffer.get_stats()
        segmenter_stats = self.volume_segmenter.get_stats()
        
        # Log every 10 seconds
        if int(current_duration) % 10 == 0:
            logger.info(f"Recording: {current_duration:.1f}s, "
                       f"Buffer: {buffer_stats['buffer_utilization']:.0%}, "
                       f"Memory: {buffer_stats['memory_usage_mb']:.1f}MB, "
                       f"Segments: {segmenter_stats['segments_detected']}")
    
    def _cleanup_resources(self) -> None:
        """Clean up audio resources."""
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
                
            if self.audio_interface:
                self.audio_interface.terminate()
                self.audio_interface = None
                
            if self.master_wave_file:
                self.master_wave_file.close()
                self.master_wave_file = None
                
        except Exception as e:
            logger.error(f"Error cleaning up audio resources: {e}")
    
    def set_segment_callback(self, callback: Callable[[SegmentInfo, np.ndarray], None]) -> None:
        """Set callback for when new segments are ready."""
        self.segment_callback = callback
        logger.info("StreamingRecorder: Segment callback set")
    
    def set_auto_stop_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for when auto-stop is triggered."""
        self.auto_stop_callback = callback
        logger.info("StreamingRecorder: Auto-stop callback set")
    
    def get_recording_stats(self) -> dict:
        """Get comprehensive recording statistics."""
        with self.lock:
            duration = 0.0
            if self.start_time:
                duration = time.time() - self.start_time
            
            buffer_stats = self.audio_buffer.get_stats()
            segmenter_stats = self.volume_segmenter.get_stats()
            vad_stats = self.vad_service.get_stats() if self.vad_service else {}
            
            return {
                'is_recording': self.is_recording,
                'duration': duration,
                'total_frames': self.total_frames_recorded,
                'master_file': self.master_file_path,
                'buffer': buffer_stats,
                'segmenter': segmenter_stats,
                'vad': vad_stats,
                'config': {
                    'sample_rate': self.config.sample_rate,
                    'channels': self.config.channels,
                    'auto_stop_enabled': self.auto_stop_enabled,
                    'auto_stop_duration': self.auto_stop_duration
                }
            }
    
    def get_real_time_display_info(self) -> dict:
        """Get information for real-time display."""
        stats = self.get_recording_stats()
        
        # Format for user display
        duration = stats['duration']
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        time_str = f"{minutes}:{seconds:02d}"
        
        buffer_pct = int(stats['buffer']['buffer_utilization'] * 100)
        memory_mb = stats['buffer']['memory_usage_mb']
        segments = stats['segmenter']['segments_detected']
        
        return {
            'time_display': time_str,
            'duration_seconds': duration,
            'buffer_percentage': buffer_pct,
            'memory_mb': memory_mb,
            'segments_detected': segments,
            'is_recording': stats['is_recording'],
            'auto_stop_enabled': self.auto_stop_enabled
        }

class StreamingDisplayHelper:
    """Helper for displaying real-time recording information."""
    
    @staticmethod
    def create_progress_bar(percentage: int, width: int = 8) -> str:
        """Create a visual progress bar."""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    @staticmethod
    def format_status_line(recorder: StreamingRecorder, show_verbose: bool = False) -> str:
        """Format a status line for display."""
        info = recorder.get_real_time_display_info()
        
        if not info['is_recording']:
            return "⏹️  Not recording"
        
        # Basic status
        progress_bar = StreamingDisplayHelper.create_progress_bar(info['buffer_percentage'])
        status = f"🎙️ {progress_bar} [{info['time_display']}]"
        
        # Add verbose information if requested
        if show_verbose:
            status += f" Buffer: {info['buffer_percentage']}% Memory: {info['memory_mb']:.1f}MB"
            
            if info['segments_detected'] > 0:
                status += f" Segments: {info['segments_detected']}"
        
        # Auto-stop indicator
        if info['auto_stop_enabled']:
            status += " (auto-stop enabled)"
        
        return status