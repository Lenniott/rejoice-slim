# src/vad_service.py

import logging
import threading
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

class VADService:
    """
    Voice Activity Detection service for analyzing audio chunks.
    
    This is a placeholder implementation that provides the foundation
    for the "Automatic Recording Termination on Prolonged Silence" feature.
    Currently logs basic audio statistics but can be extended with
    actual VAD algorithms.
    """
    
    def __init__(self, 
                 silence_threshold_chunks: int = 30,
                 chunk_duration_seconds: float = 10.0):
        """
        Initialize the VAD service.
        
        Args:
            silence_threshold_chunks: Number of consecutive silent chunks before auto-stop
            chunk_duration_seconds: Duration of each chunk in seconds
        """
        self.silence_threshold_chunks = silence_threshold_chunks
        self.chunk_duration_seconds = chunk_duration_seconds
        
        # State tracking
        self.consecutive_silent_chunks = 0
        self.total_chunks_analyzed = 0
        self.is_recording = False
        
        # Threading
        self.lock = threading.Lock()
        
        # Callback for auto-stop
        self.auto_stop_callback: Optional[Callable] = None
        
        logger.info(f"VADService initialized: {silence_threshold_chunks} chunks threshold "
                   f"({silence_threshold_chunks * chunk_duration_seconds}s)")
    
    def start_recording(self) -> None:
        """Start a new recording session."""
        with self.lock:
            self.consecutive_silent_chunks = 0
            self.total_chunks_analyzed = 0
            self.is_recording = True
        
        logger.info("VADService started recording session")
    
    def stop_recording(self) -> None:
        """Stop the current recording session."""
        with self.lock:
            self.is_recording = False
        
        logger.info(f"VADService stopped recording session. "
                   f"Analyzed {self.total_chunks_analyzed} chunks")
    
    def analyze_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Analyze an audio chunk for voice activity.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            bool: True if voice activity detected, False if silent
        """
        if not self.is_recording:
            return False
        
        with self.lock:
            self.total_chunks_analyzed += 1
            
            # Simple energy-based VAD (placeholder implementation)
            has_voice = self._detect_voice_activity(audio_chunk)
            
            if has_voice:
                self.consecutive_silent_chunks = 0
                logger.debug(f"Voice detected in chunk {self.total_chunks_analyzed}")
            else:
                self.consecutive_silent_chunks += 1
                logger.debug(f"Silent chunk {self.total_chunks_analyzed} "
                           f"(consecutive: {self.consecutive_silent_chunks})")
                
                # Check if we should trigger auto-stop
                if (self.consecutive_silent_chunks >= self.silence_threshold_chunks and
                    self.auto_stop_callback):
                    logger.info(f"Auto-stop triggered: {self.consecutive_silent_chunks} "
                              f"consecutive silent chunks")
                    self.auto_stop_callback()
            
            return has_voice
    
    def _detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """
        Simple voice activity detection based on audio energy.
        
        This is a placeholder implementation. A real VAD would use more
        sophisticated algorithms like spectral analysis, zero-crossing rate,
        or machine learning models.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            bool: True if voice activity detected
        """
        # Calculate RMS (Root Mean Square) energy
        rms_energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Simple threshold-based detection
        # This threshold would need to be tuned based on microphone sensitivity
        # and typical audio levels in the target environment
        energy_threshold = 0.01  # Adjust this value as needed
        
        # Additional check: ensure we have enough variation in the signal
        # (helps distinguish between silence and constant noise)
        signal_variance = np.var(audio_chunk)
        variance_threshold = 0.001  # Adjust this value as needed
        
        has_voice = (rms_energy > energy_threshold and 
                    signal_variance > variance_threshold)
        
        logger.debug(f"VAD analysis: RMS={rms_energy:.6f}, "
                    f"variance={signal_variance:.6f}, has_voice={has_voice}")
        
        return has_voice
    
    def set_auto_stop_callback(self, callback: Callable) -> None:
        """
        Set the callback function to be called when auto-stop is triggered.
        
        Args:
            callback: Function to call when silence threshold is reached
        """
        self.auto_stop_callback = callback
        logger.info("Auto-stop callback set")
    
    def get_stats(self) -> dict:
        """
        Get VAD service statistics.
        
        Returns:
            dict: Statistics including chunk counts, silence duration, etc.
        """
        with self.lock:
            silence_duration_seconds = (
                self.consecutive_silent_chunks * self.chunk_duration_seconds
            )
            
            return {
                'is_recording': self.is_recording,
                'total_chunks_analyzed': self.total_chunks_analyzed,
                'consecutive_silent_chunks': self.consecutive_silent_chunks,
                'silence_duration_seconds': silence_duration_seconds,
                'silence_threshold_chunks': self.silence_threshold_chunks,
                'silence_threshold_seconds': (
                    self.silence_threshold_chunks * self.chunk_duration_seconds
                )
            }
    
    def reset(self) -> None:
        """Reset the VAD service state."""
        with self.lock:
            self.consecutive_silent_chunks = 0
            self.total_chunks_analyzed = 0
            self.is_recording = False
        
        logger.info("VADService reset")
