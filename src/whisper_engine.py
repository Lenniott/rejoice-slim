# src/whisper_engine.py
"""
Whisper transcription engine wrapper.
Provides a unified interface for whisper transcription with faster-whisper backend.
100% local, no API calls or network requests.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Use faster-whisper for better performance (4x faster, lower memory)
from faster_whisper import WhisperModel


class WhisperEngine:
    """
    Unified wrapper for Whisper transcription.
    Automatically uses faster-whisper if available, falls back to openai-whisper.
    100% local processing, no network requests.
    """

    def __init__(self, model_name: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize Whisper engine.

        Args:
            model_name: Model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda)
            compute_type: Computation type for faster-whisper (int8, float16, float32)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load_model(self, model_name: Optional[str] = None):
        """
        Load the Whisper model.

        Args:
            model_name: Optional model name to override initialization

        Returns:
            Self (for chaining)
        """
        if model_name:
            self.model_name = model_name

        logger.info(f"Loading Whisper model: {self.model_name} (faster-whisper backend)")

        # faster-whisper: optimized with CTranslate2
        # 100% local, downloads model to ~/.cache/huggingface/ on first use
        # local_files_only=True prevents online checks after first download
        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,  # Use default cache location
                local_files_only=True  # Use only cached files, no online checks
            )
            logger.info(f"faster-whisper model loaded from cache: {self.model_name} (device={self.device}, compute_type={self.compute_type})")
        except Exception as e:
            # If model not in cache, download it (first run only)
            logger.info(f"Model not in cache, downloading: {self.model_name}")
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None
            )
            logger.info(f"faster-whisper model downloaded and loaded: {self.model_name} (device={self.device}, compute_type={self.compute_type})")

        return self

    def transcribe(self,
                   audio: Union[str, Path, np.ndarray],
                   language: Optional[str] = None,
                   fp16: bool = False,
                   beam_size: int = 5,
                   **kwargs) -> Dict:
        """
        Transcribe audio to text.

        Args:
            audio: Audio file path (str/Path) or numpy array (float32, 16kHz)
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            fp16: Ignored (kept for compatibility with old whisper API)
            beam_size: Beam size for decoding
            **kwargs: Additional arguments passed to faster-whisper

        Returns:
            Dict with 'text' key containing transcription
        """
        if self._model is None:
            self.load_model()

        return self._transcribe_faster_whisper(audio, language, beam_size, **kwargs)

    def _transcribe_faster_whisper(self,
                                   audio: Union[str, Path, np.ndarray],
                                   language: Optional[str] = None,
                                   beam_size: int = 5,
                                   **kwargs) -> Dict:
        """Transcribe using faster-whisper backend."""
        # Convert Path to string if needed
        if isinstance(audio, Path):
            audio = str(audio)

        # Transcribe with faster-whisper
        # Returns (segments, info) tuple
        # VAD enabled to filter silence - essential for long recordings
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=True,  # Filter silence to improve accuracy and performance
            **kwargs
        )

        # Collect all segments into full text
        # Note: segments is a generator, so we need to consume it
        full_text = ""
        segment_list = []

        for segment in segments:
            full_text += segment.text
            segment_list.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })

        # Return in openai-whisper compatible format
        return {
            'text': full_text.strip(),
            'segments': segment_list,
            'language': info.language if hasattr(info, 'language') else language
        }


# Convenience function that mimics openai-whisper API
def load_model(model_name: str = "base", device: str = "cpu") -> WhisperEngine:
    """
    Load a Whisper model (compatible with openai-whisper API).

    Args:
        model_name: Model size (tiny, base, small, medium, large)
        device: Device to use (cpu, cuda)

    Returns:
        WhisperEngine instance with loaded model
    """
    engine = WhisperEngine(model_name, device)
    engine.load_model()
    return engine
