"""
File management utilities for transcript operations.
Handles creation, lookup, and management of transcript files.
"""

import os
import re
from datetime import datetime
from typing import Optional, Tuple, List, Callable
from pathlib import Path
from id_generator import TranscriptIDGenerator
from file_header import TranscriptHeader
from audio_manager import AudioFileManager
import string


class TranscriptFileManager:
    """Manages transcript file operations with ID-based naming."""
    
    def __init__(self, save_path: str, output_format: str = "md"):
        """
        Initialize file manager.
        
        Args:
            save_path: Directory where transcript files are saved
            output_format: Default output format ("md" or "txt")
        """
        self.save_path = save_path
        self.output_format = output_format
        self.id_generator = TranscriptIDGenerator(save_path)
        self.audio_manager = AudioFileManager(save_path)
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
    
    def create_new_transcript(self, transcript_text: str, 
                            generated_filename: str = "transcript",
                            custom_id: Optional[str] = None,
                            session_audio_file: Optional[Path] = None) -> Tuple[str, str, Optional[str]]:
        """
        Create a new transcript file with the new naming format and optionally store associated audio.
        Format: {id}_DDMMYYYY_{generated_filename}.{ext}
        
        Args:
            transcript_text: Transcribed text content
            generated_filename: Generated filename (from AI or default)
            custom_id: Optional custom ID (must be unique positive integer)
            session_audio_file: Optional path to session audio file to store permanently
            
        Returns:
            tuple: (file_path, transcript_id, stored_audio_path)
                - file_path: Path to the created transcript file
                - transcript_id: The transcript ID
                - stored_audio_path: Path to stored audio file (None if not stored)
            
        Raises:
            ValueError: If custom_id is invalid or already exists
        """
        # Generate or validate ID
        if custom_id:
            if not self._is_valid_custom_id(custom_id):
                raise ValueError(f"Invalid custom ID: {custom_id}")
            if self.id_generator.id_exists(custom_id):
                raise ValueError(f"ID already exists: {custom_id}")
            transcript_id = custom_id
        else:
            transcript_id = self.id_generator.get_next_id()
        
        # Create header and content
        header = TranscriptHeader(transcript_id)
        file_content = header.create_file_content(transcript_text, self.output_format)
        
        # Generate filename with new format: ID_DDMMYYYY_name.ext
        date_str = datetime.now().strftime("%d%m%Y")
        clean_filename = self._clean_filename(generated_filename)
        filename = f"{transcript_id}_{date_str}_{clean_filename}.{self.output_format}"
        file_path = os.path.join(self.save_path, filename)
        
        # Write transcript file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        # Store audio file if provided
        audio_file_path = None
        if session_audio_file and session_audio_file.exists():
            audio_file_path = self.audio_manager.store_session_audio(
                session_audio_file, transcript_id, clean_filename, 1
            )
            if audio_file_path:
                print(f"🎵 Audio stored: {os.path.basename(audio_file_path)}")
            else:
                print("⚠️ Failed to store audio file")
        
        return file_path, transcript_id, audio_file_path
    
    def find_transcript(self, id_reference: str) -> Optional[str]:
        """
        Find transcript file by ID reference using flexible matching.
        
        Args:
            id_reference: ID reference (e.g., "123456", "-123456")
            
        Returns:
            str or None: Full path to transcript file, or None if not found
        
        Raises:
            ValueError: If multiple files match the same ID
        """
        # Parse the reference to get clean ID
        clean_id = self.id_generator.parse_reference_id(id_reference)
        if not clean_id:
            return None
        
        try:
            return self.id_generator.find_transcript_by_id(clean_id)
        except ValueError as e:
            # Re-raise with more context about the ID reference
            raise ValueError(f"ID conflict for reference '{id_reference}': {str(e)}")  from e
    
    def append_to_transcript(self, id_reference: str, 
                           additional_text: str,
                           regenerate_metadata: bool = False,
                           metadata_callback: Optional[Callable] = None,
                           session_audio_file: Optional[Path] = None) -> Optional[str]:
        """
        Append text to an existing transcript and optionally store additional audio.
        
        Args:
            id_reference: ID reference to find transcript
            additional_text: Text to append
            regenerate_metadata: Whether to regenerate AI metadata after appending
            metadata_callback: Optional callback function to get new metadata
            session_audio_file: Optional path to session audio file to store permanently
            
        Returns:
            str or None: Path to updated file, or None if not found
        """
        file_path = self.find_transcript(id_reference)
        if not file_path:
            return None
        
        try:
            # Read existing content
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # Extract current transcript content (without header)
            current_transcript = TranscriptHeader.extract_transcript_content(existing_content)
            
            # Combine with new text
            combined_transcript = f"{current_transcript}\n\n{additional_text}"
            
            # Get ID from header and recreate file with combined content
            header_data = TranscriptHeader.parse_header(existing_content)
            if header_data and 'id' in header_data:
                transcript_id = str(header_data['id'])
                
                # Store additional audio file if provided
                if session_audio_file and session_audio_file.exists():
                    # Get the base filename for the audio
                    current_filename = os.path.basename(file_path)
                    # Extract the generated filename part from transcript filename
                    import re
                    pattern = re.compile(r'^(\d+)_(\d{8})_(.*)\.(.+)$')
                    match = pattern.match(current_filename)
                    if match:
                        base_filename = match.group(3)
                    else:
                        base_filename = "recording"
                    
                    # Get next sequence number for this transcript
                    sequence_num = self.audio_manager.get_next_sequence_number(transcript_id, base_filename)
                    
                    audio_file_path = self.audio_manager.store_session_audio(
                        session_audio_file, transcript_id, base_filename, sequence_num
                    )
                    if audio_file_path:
                        print(f"🎵 Additional audio stored: {os.path.basename(audio_file_path)}")
                    else:
                        print("⚠️ Failed to store additional audio file")
                
                # Create new header with original creation date
                creation_date = None
                if 'created' in header_data:
                    try:
                        creation_date = datetime.fromisoformat(header_data['created'])
                    except (ValueError, TypeError):
                        pass
                
                # Regenerate metadata if requested and callback provided
                if regenerate_metadata and metadata_callback:
                    try:
                        new_metadata = metadata_callback(combined_transcript)
                        if new_metadata and new_metadata.get('filename'):
                            # Update filename with new AI-generated name if different
                            new_filename = new_metadata['filename']
                            current_filename = os.path.basename(file_path)
                            
                            # Extract date and ID from current filename
                            import re
                            pattern = re.compile(r'^(\d+)_(\d{8})_.*\.(txt|md)$')
                            match = pattern.match(current_filename)
                            if match:
                                file_id = match.group(1)
                                date_str = match.group(2)
                                ext = match.group(3)
                                
                                # Create new filename with updated name
                                clean_filename = self._clean_filename(new_filename)
                                new_filename_full = f"{file_id}_{date_str}_{clean_filename}.{ext}"
                                new_file_path = os.path.join(self.save_path, new_filename_full)
                                
                                # Rename file if the name changed
                                if new_filename_full != current_filename:
                                    os.rename(file_path, new_file_path)
                                    file_path = new_file_path
                    except Exception as e:
                        # Don't fail the append if metadata regeneration fails
                        pass
                
                header = TranscriptHeader(transcript_id, creation_date)
                new_content = header.create_file_content(combined_transcript, self.output_format)
                
                # Write updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                return file_path
        
        except (IOError, OSError):
            pass
        
        return None
    
    def list_transcripts(self) -> List[Tuple[str, str, datetime]]:
        """
        List all ID-format transcripts with metadata.
        
        Returns:
            list: List of (id, filename, creation_date) tuples
        """
        transcripts = []
        
        try:
            for filename in os.listdir(self.save_path):
                if TranscriptHeader.is_id_format_file(filename):
                    file_path = os.path.join(self.save_path, filename)
                    
                    # Extract ID from filename pattern: *_DDMMYYYY_000001.ext
                    transcript_id = self._extract_id_from_filename(filename)
                    if transcript_id:
                        # Get creation date from header
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            header_data = TranscriptHeader.parse_header(content)
                            creation_date = datetime.now()  # default
                            
                            if header_data and 'created' in header_data:
                                try:
                                    creation_date = datetime.fromisoformat(header_data['created'])
                                except (ValueError, TypeError):
                                    pass
                            
                            transcripts.append((transcript_id, filename, creation_date))
                        
                        except (IOError, OSError):
                            continue
        
        except OSError:
            pass
        
        # Sort by creation date (newest first)
        transcripts.sort(key=lambda x: x[2], reverse=True)
        return transcripts
    
    def _extract_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract ID from filename with format: ID_DDMMYYYY_*.ext
        
        Args:
            filename: Filename to extract ID from
            
        Returns:
            str or None: Extracted ID or None if not found
        """
        pattern = re.compile(r'^(\d+)_\d{8}_.*\.(txt|md)$')
        match = pattern.match(filename)
        if match:
            return match.group(1)
        return None
    
    def get_transcript_content(self, id_reference: str) -> Optional[str]:
        """
        Get the transcript content (without header) by ID.
        
        Args:
            id_reference: ID reference to find transcript
            
        Returns:
            str or None: Transcript content, or None if not found
        
        Raises:
            ValueError: If multiple files match the same ID
        """
        file_path = self.find_transcript(id_reference)  # This can raise ValueError
        if not file_path:
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return TranscriptHeader.extract_transcript_content(content)
        
        except (IOError, OSError):
            return None
    
    def get_audio_files_for_transcript(self, id_reference: str) -> List[dict]:
        """
        Get information about audio files associated with a transcript.
        
        Args:
            id_reference: ID reference to find transcript
            
        Returns:
            List[dict]: List of audio file information dictionaries
        """
        # Parse the reference to get clean ID
        clean_id = self.id_generator.parse_reference_id(id_reference)
        if not clean_id:
            return []
        
        audio_files = self.audio_manager.find_audio_files_for_transcript(clean_id)
        return [self.audio_manager.get_audio_info(af) for af in audio_files]
    
    def list_transcripts_with_audio(self) -> List[Tuple[str, str, datetime, int, float]]:
        """
        List all ID-format transcripts with metadata including audio file count and total duration.
        
        Returns:
            list: List of (id, filename, creation_date, audio_count, total_audio_duration) tuples
        """
        transcripts = []
        
        try:
            for filename in os.listdir(self.save_path):
                if TranscriptHeader.is_id_format_file(filename):
                    file_path = os.path.join(self.save_path, filename)
                    
                    # Extract ID from filename pattern: ID_DDMMYYYY_*.ext
                    transcript_id = self._extract_id_from_filename(filename)
                    if transcript_id:
                        # Get creation date from header
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            header_data = TranscriptHeader.parse_header(content)
                            creation_date = datetime.now()  # default
                            
                            if header_data and 'created' in header_data:
                                try:
                                    creation_date = datetime.fromisoformat(header_data['created'])
                                except (ValueError, TypeError):
                                    pass
                            
                            # Get audio file information
                            audio_files = self.audio_manager.find_audio_files_for_transcript(transcript_id)
                            audio_count = len(audio_files)
                            total_duration = 0.0
                            
                            for audio_file in audio_files:
                                audio_info = self.audio_manager.get_audio_info(audio_file)
                                total_duration += audio_info.get('duration', 0.0)
                            
                            transcripts.append((transcript_id, filename, creation_date, audio_count, total_duration))
                        
                        except (IOError, OSError):
                            continue
        
        except OSError:
            pass
        
        # Sort by creation date (newest first)
        transcripts.sort(key=lambda x: x[2], reverse=True)
        return transcripts
    
    def update_transcript_content(self, file_path: str, transcript_text: str) -> bool:
        """
        Update an existing transcript file with new content while preserving header metadata.
        
        Args:
            file_path: Path to the transcript file
            transcript_text: New transcription text to write
        
        Returns:
            bool: True if update succeeded, False otherwise
        """
        if not file_path or not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            header_data = TranscriptHeader.parse_header(content)
            if not header_data or 'id' not in header_data:
                return False
            
            transcript_id = str(header_data['id'])
            creation_date = None
            if 'created' in header_data:
                try:
                    from datetime import datetime
                    creation_date = datetime.fromisoformat(header_data['created'])
                except (ValueError, TypeError):
                    pass
            
            header = TranscriptHeader(transcript_id, creation_date)
            new_content = header.create_file_content(transcript_text.strip(), self.output_format)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True
        except Exception:
            return False
    
    def update_transcript_status(self, file_path: str, status: str) -> bool:
        """
        Update the status field in a transcript's YAML frontmatter.
        
        Args:
            file_path: Path to the transcript file
            status: New status value (e.g., "raw", "processed")
        
        Returns:
            bool: True if update succeeded, False otherwise
        """
        if not file_path or not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            header_data = TranscriptHeader.parse_header(content)
            if not header_data:
                return False
            
            # Update status in header
            header_data['status'] = status
            
            # Regenerate YAML
            import yaml
            yaml_content = yaml.dump(header_data, default_flow_style=False, sort_keys=False)
            
            # Replace old frontmatter with updated one
            import re
            yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.MULTILINE | re.DOTALL)
            content_after_frontmatter = yaml_pattern.sub('', content, count=1)
            
            new_content = f"---\n{yaml_content}---\n{content_after_frontmatter}"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True
        except Exception:
            return False
    
    def _is_valid_custom_id(self, custom_id: str) -> bool:
        """
        Validate custom ID format.
        
        Args:
            custom_id: Custom ID to validate
            
        Returns:
            bool: True if valid format
        """
        if not custom_id or not custom_id.isdigit():
            return False
        
        # Check that it's a positive integer
        num = int(custom_id)
        return num >= 1
    
    def reprocess_transcript_audio(self, id_reference: str, 
                                 transcription_callback: Optional[Callable] = None,
                                 summarization_callback: Optional[Callable] = None,
                                 overwrite_existing: bool = False) -> Tuple[bool, str, List[str]]:
        """
        Reprocess all audio files for a transcript ID - transcribe and optionally summarize.
        
        Args:
            id_reference: ID reference to find transcript and audio files
            transcription_callback: Function to transcribe audio files (audio_path) -> transcript_text
            summarization_callback: Function to generate summary/metadata (transcript_text) -> metadata_dict
            overwrite_existing: Whether to overwrite existing transcript or create new one
            
        Returns:
            Tuple[bool, str, List[str]]: (success, transcript_path, processed_audio_files)
        """
        # Parse the reference to get clean ID
        clean_id = self.id_generator.parse_reference_id(id_reference)
        if not clean_id:
            return False, "", []
        
        # Find all audio files for this transcript ID
        audio_files = self.audio_manager.find_audio_files_for_transcript(clean_id)
        if not audio_files:
            return False, f"No audio files found for transcript ID {clean_id}", []
        
        print(f"🎵 Found {len(audio_files)} audio files for transcript {clean_id}")
        
        # Process each audio file
        transcribed_segments = []
        processed_files = []
        
        for i, audio_file in enumerate(sorted(audio_files), 1):
            print(f"🔄 Processing audio file {i}/{len(audio_files)}: {os.path.basename(audio_file)}")
            
            if transcription_callback:
                try:
                    # Transcribe this audio file
                    segment_text = transcription_callback(str(audio_file))
                    if segment_text and segment_text.strip():
                        transcribed_segments.append(f"=== Audio Segment {i} ({os.path.basename(audio_file)}) ===\n\n{segment_text.strip()}")
                        processed_files.append(os.path.basename(audio_file))
                        print(f"✅ Transcribed: {os.path.basename(audio_file)}")
                    else:
                        print(f"⚠️ Empty transcription: {os.path.basename(audio_file)}")
                except Exception as e:
                    print(f"❌ Failed to transcribe {os.path.basename(audio_file)}: {str(e)}")
                    continue
            else:
                print(f"⚠️ No transcription callback provided, skipping {os.path.basename(audio_file)}")
        
        if not transcribed_segments:
            return False, "No audio files were successfully transcribed", []
        
        # Combine all transcribed segments
        combined_transcript = "\n\n".join(transcribed_segments)
        
        # Generate filename and metadata if summarization callback provided
        generated_filename = f"reprocessed_transcript"
        if summarization_callback:
            try:
                print("🤖 Generating AI summary and filename...")
                metadata = summarization_callback(combined_transcript)
                if metadata and metadata.get('filename'):
                    generated_filename = metadata['filename']
                    print(f"🎯 AI-generated filename: {generated_filename}")
            except Exception as e:
                print(f"⚠️ AI summarization failed, using default filename: {str(e)}")
        
        # Handle existing transcript
        transcript_path = None
        if overwrite_existing:
            # Find and update existing transcript
            try:
                existing_path = self.find_transcript(id_reference)
            except ValueError:
                existing_path = None
                
            if existing_path:
                try:
                    # Read existing content to preserve header data
                    with open(existing_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    
                    header_data = TranscriptHeader.parse_header(existing_content)
                    creation_date = None
                    if header_data and 'created' in header_data:
                        try:
                            creation_date = datetime.fromisoformat(header_data['created'])
                        except (ValueError, TypeError):
                            pass
                    
                    # Create updated content with same ID and creation date
                    header = TranscriptHeader(clean_id, creation_date)
                    new_content = header.create_file_content(combined_transcript, self.output_format)
                    
                    # Write updated content
                    with open(existing_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    transcript_path = existing_path
                    print(f"✅ Updated existing transcript: {os.path.basename(existing_path)}")
                    
                except Exception as e:
                    print(f"❌ Failed to update existing transcript: {str(e)}")
                    # Fall back to creating new file
                    overwrite_existing = False
        
        # Create new transcript if not overwriting or overwrite failed
        if not overwrite_existing or not transcript_path:
            try:
                # Create new transcript
                transcript_path, new_id, _ = self.create_new_transcript(
                    combined_transcript, 
                    generated_filename
                )
                print(f"✅ Created new transcript: {os.path.basename(transcript_path)} (ID: {new_id})")
                
            except Exception as e:
                return False, f"Failed to create transcript: {str(e)}", processed_files
        
        return True, transcript_path, processed_files
    
    def reprocess_all_failed_transcripts(self, 
                                       transcription_callback: Optional[Callable] = None,
                                       summarization_callback: Optional[Callable] = None) -> List[Tuple[str, bool, str]]:
        """
        Find and reprocess all transcript IDs that have audio files but no transcript content.
        
        Args:
            transcription_callback: Function to transcribe audio files
            summarization_callback: Function to generate summary/metadata
            
        Returns:
            List[Tuple[str, bool, str]]: List of (transcript_id, success, message) tuples
        """
        # Get all audio files and extract unique transcript IDs
        audio_transcript_ids = set()
        try:
            audio_dir = os.path.join(self.save_path, "audio")
            if os.path.exists(audio_dir):
                for audio_file in os.listdir(audio_dir):
                    if audio_file.endswith('.wav'):
                        # Extract ID from audio filename: ID_DDMMYYYY_*.wav
                        pattern = re.compile(r'^(\d+)_\d{8}_.*\.wav$')
                        match = pattern.match(audio_file)
                        if match:
                            audio_transcript_ids.add(match.group(1))
        except OSError:
            return []
        
        # Get all existing transcript IDs
        existing_transcript_ids = set()
        for transcript_id, _, _ in self.list_transcripts():
            existing_transcript_ids.add(transcript_id)
        
        # Find IDs with audio but no transcript
        orphaned_audio_ids = audio_transcript_ids - existing_transcript_ids
        
        if not orphaned_audio_ids:
            print("✅ No orphaned audio files found (all audio has corresponding transcripts)")
            return []
        
        print(f"🔄 Found {len(orphaned_audio_ids)} transcript IDs with audio but no transcript file")
        
        # Reprocess each orphaned ID
        results = []
        for transcript_id in sorted(orphaned_audio_ids):
            print(f"\n🎵 Reprocessing orphaned audio for ID {transcript_id}...")
            
            success, path, files = self.reprocess_transcript_audio(
                transcript_id, transcription_callback, summarization_callback, overwrite_existing=False
            )
            
            if success:
                message = f"Successfully created transcript from {len(files)} audio files"
                print(f"✅ {message}")
            else:
                message = f"Failed: {path}"
                print(f"❌ {message}")
            
            results.append((transcript_id, success, message))
        
        return results

    def _clean_filename(self, filename: str) -> str:
        """Clean filename to make it filesystem-safe."""
        # Remove or replace invalid characters
        valid_chars = f"-_.{string.ascii_letters}{string.digits}"
        cleaned = ''.join(c if c in valid_chars else '_' for c in filename)
        
        # Remove multiple underscores and trim
        cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        
        # Ensure it's not empty and not too long
        if not cleaned:
            cleaned = "transcript"
        elif len(cleaned) > 50:
            cleaned = cleaned[:50].strip('_')
        
        return cleaned