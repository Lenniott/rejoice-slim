"""
File management utilities for transcript operations.
Handles creation, lookup, and management of transcript files.
"""

import os
import re
from datetime import datetime
from typing import Optional, Tuple, List, Callable
from id_generator import TranscriptIDGenerator
from file_header import TranscriptHeader
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
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
    
    def create_new_transcript(self, transcript_text: str, 
                            generated_filename: str = "transcript",
                            custom_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Create a new transcript file with the new naming format.
        Format: {generated_filename}_DDMMYYYY_{id}.{ext}
        
        Args:
            transcript_text: Transcribed text content
            generated_filename: Generated filename (from AI or default)
            custom_id: Optional custom ID (must be unique positive integer)
            
        Returns:
            tuple: (file_path, transcript_id)
            
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
        
        # Generate filename with new format: name_DDMMYYYY_ID.ext
        from datetime import datetime
        date_str = datetime.now().strftime("%d%m%Y")
        clean_filename = self._clean_filename(generated_filename)
        filename = f"{clean_filename}_{date_str}_{transcript_id}.{self.output_format}"
        file_path = os.path.join(self.save_path, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        return file_path, transcript_id
    
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
                           metadata_callback: Optional[Callable] = None) -> Optional[str]:
        """
        Append text to an existing transcript.
        
        Args:
            id_reference: ID reference to find transcript
            additional_text: Text to append
            regenerate_metadata: Whether to regenerate AI metadata after appending
            metadata_callback: Optional callback function to get new metadata
            
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
                            pattern = re.compile(r'^.*_(\d{8})_(\d+)\.(txt|md)$')
                            match = pattern.match(current_filename)
                            if match:
                                date_str = match.group(1)
                                file_id = match.group(2)
                                ext = match.group(3)
                                
                                # Create new filename with updated name
                                clean_filename = self._clean_filename(new_filename)
                                new_filename_full = f"{clean_filename}_{date_str}_{file_id}.{ext}"
                                new_file_path = os.path.join(self.save_path, new_filename_full)
                                
                                # Rename file if the name changed
                                if new_filename_full != current_filename:
                                    import os
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
        Extract ID from filename with format: *_DDMMYYYY_ID.ext
        
        Args:
            filename: Filename to extract ID from
            
        Returns:
            str or None: Extracted ID or None if not found
        """
        pattern = re.compile(r'^.*_\d{8}_(\d+)\.(txt|md)$')
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