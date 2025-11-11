"""
Audio file management utilities for transcript operations.
Handles permanent storage and linking of audio files with transcripts.
"""

import os
import shutil
import wave
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioFileManager:
    """Manages audio file operations with ID-based naming and transcript linking."""
    
    def __init__(self, save_path: str):
        """
        Initialize audio file manager.
        
        Args:
            save_path: Directory where transcript files are saved (audio will be stored in subdirectory)
        """
        self.save_path = save_path
        self.audio_path = os.path.join(save_path, "audio")
        
        # Ensure audio directory exists
        os.makedirs(self.audio_path, exist_ok=True)
    
    def store_session_audio(self, session_audio_file: Path, transcript_id: str, 
                           generated_filename: str = "recording", 
                           sequence_num: int = 1) -> Optional[str]:
        """
        Move session audio file to permanent storage with ID-based naming.
        
        Args:
            session_audio_file: Temporary session audio file path
            transcript_id: ID of the associated transcript
            generated_filename: Generated filename (from AI or default)
            sequence_num: Sequence number for multiple recordings with same ID
            
        Returns:
            str or None: Path to stored audio file, or None if failed
        """
        try:
            if not session_audio_file.exists():
                logger.warning(f"Session audio file does not exist: {session_audio_file}")
                return None
            
            # Create filename with new format: ID_DDMMYYYY_name_seq.wav
            date_str = datetime.now().strftime("%d%m%Y")
            clean_filename = self._clean_filename(generated_filename)
            
            if sequence_num == 1:
                audio_filename = f"{transcript_id}_{date_str}_{clean_filename}.wav"
            else:
                audio_filename = f"{transcript_id}_{date_str}_{clean_filename}_{sequence_num:03d}.wav"
            
            audio_file_path = os.path.join(self.audio_path, audio_filename)
            
            # Move session file to permanent storage
            shutil.move(str(session_audio_file), audio_file_path)
            
            # Verify the file was moved successfully
            if os.path.exists(audio_file_path):
                file_size = os.path.getsize(audio_file_path)
                duration = self._get_audio_duration(audio_file_path)
                
                logger.info(f"Audio stored: {audio_filename} ({duration:.1f}s, {file_size/1024/1024:.1f}MB)")
                return audio_file_path
            else:
                logger.error(f"Failed to store audio file: {audio_filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing session audio: {e}")
            return None
    
    def find_audio_files_for_transcript(self, transcript_id: str) -> List[str]:
        """
        Find all audio files associated with a transcript ID.
        
        Args:
            transcript_id: Transcript ID to find audio files for
            
        Returns:
            List[str]: List of audio file paths for this transcript
        """
        audio_files = []
        
        try:
            if not os.path.exists(self.audio_path):
                return audio_files
            
            # Pattern to match: ID_DDMMYYYY_*.wav
            for filename in os.listdir(self.audio_path):
                if filename.endswith('.wav'):
                    # Check if filename starts with transcript_id_
                    if filename.startswith(f"{transcript_id}_"):
                        # Verify it matches the full pattern
                        parts = filename.split('_')
                        if len(parts) >= 3 and parts[0] == transcript_id:
                            # parts[1] should be date (8 digits)
                            if len(parts[1]) == 8 and parts[1].isdigit():
                                audio_files.append(os.path.join(self.audio_path, filename))
            
            # Sort by filename to get chronological order
            audio_files.sort()
            
        except Exception as e:
            logger.error(f"Error finding audio files for transcript {transcript_id}: {e}")
        
        return audio_files
    
    def get_next_sequence_number(self, transcript_id: str, generated_filename: str) -> int:
        """
        Get the next sequence number for multiple recordings with the same transcript ID.
        
        Args:
            transcript_id: Transcript ID
            generated_filename: Generated filename
            
        Returns:
            int: Next available sequence number
        """
        try:
            date_str = datetime.now().strftime("%d%m%Y")
            clean_filename = self._clean_filename(generated_filename)
            base_pattern = f"{transcript_id}_{date_str}_{clean_filename}"
            
            max_sequence = 0
            
            for filename in os.listdir(self.audio_path):
                if filename.startswith(base_pattern) and filename.endswith('.wav'):
                    if filename == f"{base_pattern}.wav":
                        max_sequence = max(max_sequence, 1)
                    elif filename.startswith(f"{base_pattern}_") and filename.endswith('.wav'):
                        # Extract sequence number
                        try:
                            seq_part = filename[len(f"{base_pattern}_"):-4]  # Remove .wav
                            if seq_part.isdigit():
                                max_sequence = max(max_sequence, int(seq_part))
                        except (ValueError, IndexError):
                            pass
            
            return max_sequence + 1
            
        except Exception as e:
            logger.error(f"Error getting next sequence number: {e}")
            return 1
    
    def get_audio_info(self, audio_file_path: str) -> dict:
        """
        Get information about an audio file.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            dict: Audio file information (duration, size, format, etc.)
        """
        info = {
            'path': audio_file_path,
            'exists': False,
            'duration': 0.0,
            'size_bytes': 0,
            'size_mb': 0.0,
            'format': 'unknown'
        }
        
        try:
            if os.path.exists(audio_file_path):
                info['exists'] = True
                info['size_bytes'] = os.path.getsize(audio_file_path)
                info['size_mb'] = info['size_bytes'] / (1024 * 1024)
                info['duration'] = self._get_audio_duration(audio_file_path)
                info['format'] = 'WAV' if audio_file_path.lower().endswith('.wav') else 'unknown'
        
        except Exception as e:
            logger.error(f"Error getting audio info for {audio_file_path}: {e}")
        
        return info
    
    def list_all_audio_files(self) -> List[Tuple[str, str, dict]]:
        """
        List all audio files with their associated transcript IDs.
        
        Returns:
            List[Tuple[str, str, dict]]: List of (transcript_id, filename, info) tuples
        """
        audio_files = []
        
        try:
            if not os.path.exists(self.audio_path):
                return audio_files
            
            for filename in os.listdir(self.audio_path):
                if filename.endswith('.wav'):
                    # Extract transcript ID from filename
                    transcript_id = self._extract_id_from_filename(filename)
                    if transcript_id:
                        file_path = os.path.join(self.audio_path, filename)
                        info = self.get_audio_info(file_path)
                        audio_files.append((transcript_id, filename, info))
            
            # Sort by transcript ID and then by filename
            audio_files.sort(key=lambda x: (int(x[0]), x[1]))
            
        except Exception as e:
            logger.error(f"Error listing audio files: {e}")
        
        return audio_files
    
    def delete_audio_file(self, audio_file_path: str) -> bool:
        """
        Delete an audio file (use with caution).
        
        Args:
            audio_file_path: Path to audio file to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                logger.info(f"Audio file deleted: {audio_file_path}")
                return True
            else:
                logger.warning(f"Audio file not found for deletion: {audio_file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting audio file {audio_file_path}: {e}")
            return False
    
    def _get_audio_duration(self, audio_file_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            float: Duration in seconds, or 0.0 if unable to determine
        """
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                return duration
        except Exception as e:
            logger.debug(f"Could not get duration for {audio_file_path}: {e}")
            return 0.0
    
    def _extract_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract transcript ID from audio filename.
        
        Args:
            filename: Audio filename to extract ID from
            
        Returns:
            str or None: Extracted ID or None if not found
        """
        try:
            # Pattern: ID_DDMMYYYY_*.wav
            parts = filename.split('_')
            if len(parts) >= 3 and filename.endswith('.wav'):
                transcript_id = parts[0]
                date_part = parts[1]
                # Validate ID is numeric and date part is 8 digits
                if transcript_id.isdigit() and len(date_part) == 8 and date_part.isdigit():
                    return transcript_id
        except Exception:
            pass
        
        return None
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename to make it filesystem-safe."""
        import string
        import re
        
        # Remove or replace invalid characters
        valid_chars = f"-_.{string.ascii_letters}{string.digits}"
        cleaned = ''.join(c if c in valid_chars else '_' for c in filename)
        
        # Remove multiple underscores and trim
        cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        
        # Ensure it's not empty and not too long
        if not cleaned:
            cleaned = "recording"
        elif len(cleaned) > 50:
            cleaned = cleaned[:50].strip('_')
        
        return cleaned