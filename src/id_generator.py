"""
ID generation utilities for transcript files.
Provides unique sequential IDs for transcript filenames.
"""

import os
import glob
import re
import threading
from typing import Optional


class TranscriptIDGenerator:
    """Generates unique sequential IDs for transcript files."""
    
    def __init__(self, save_path: str):
        """
        Initialize ID generator.
        
        Args:
            save_path: Directory where transcript files are saved
        """
        self.save_path = save_path
        self._lock = threading.Lock()
    
    def get_next_id(self) -> str:
        """
        Generate the next available sequential unique ID.
        
        Returns:
            str: Sequential unique ID (e.g., "1", "2", "42", "123456")
        """
        with self._lock:
            highest_id = self._scan_existing_ids()
            next_id = highest_id + 1
            return str(next_id)
    
    def _scan_existing_ids(self) -> int:
        """
        Scan directory for existing transcript files with the new naming format.
        Format: {id}_DDMMYYYY_{generated_name}.{ext}
        
        Returns:
            int: Highest existing ID number (0 if none found)
        """
        if not os.path.exists(self.save_path):
            return 0
        
        # Pattern to match new format: [digits]_DDMMYYYY_*.txt/md (any number of digits)
        id_pattern = re.compile(r'^(\d+)_\d{8}_.*\.(txt|md)$')
        
        highest_id = 0
        
        try:
            # Scan all files in save directory
            for filename in os.listdir(self.save_path):
                match = id_pattern.match(filename)
                if match:
                    file_id = int(match.group(1))
                    highest_id = max(highest_id, file_id)
        except OSError:
            # Directory access error, return 0 to start from beginning
            return 0
        
        return highest_id
    
    def id_exists(self, id_str: str) -> bool:
        """
        Check if a transcript with the given ID already exists.
        
        Args:
            id_str: ID string to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        if not self._is_valid_id(id_str):
            return False
        
        # Convert to integer and back to handle zero-padding differences
        id_num = int(id_str)
        
        # Look for files that match this ID numerically
        id_pattern = re.compile(r'^(\d+)_\d{8}_.*\.(txt|md)$')
        
        try:
            for filename in os.listdir(self.save_path):
                match = id_pattern.match(filename)
                if match:
                    file_id_num = int(match.group(1))
                    if file_id_num == id_num:
                        return True
        except OSError:
            pass
        
        return False
    
    def find_transcript_by_id(self, id_str: str) -> Optional[str]:
        """
        Find transcript file by ID using flexible matching.
        Looks for files where the first part before the first underscore matches the ID.
        
        Args:
            id_str: ID string to find
            
        Returns:
            str or None: Full path to transcript file, or None if not found
        
        Raises:
            ValueError: If multiple files match the same ID
        """
        if not self._is_valid_id(id_str):
            return None
        
        # Convert to integer to handle zero-padding differences
        id_num = int(id_str)
        
        # Flexible pattern: any filename starting with ID_
        # This matches both new format (ID_DDMMYYYY_name.md) and any other format with ID_ at the start
        id_pattern = re.compile(r'^(\d+)_.*\.(txt|md)$')
        
        matching_files = []
        
        try:
            for filename in os.listdir(self.save_path):
                match = id_pattern.match(filename)
                if match:
                    file_id_num = int(match.group(1))
                    if file_id_num == id_num:
                        matching_files.append(filename)
        except OSError:
            return None
        
        if len(matching_files) == 0:
            return None
        elif len(matching_files) == 1:
            return os.path.join(self.save_path, matching_files[0])
        else:
            # Multiple files match - this is an error condition
            raise ValueError(f"Multiple files found with ID {id_num}: {', '.join(matching_files)}")
        
        return None
    
    def _is_valid_id(self, id_str: str) -> bool:
        """
        Validate ID format.
        
        Args:
            id_str: ID string to validate
            
        Returns:
            bool: True if valid positive integer ID
        """
        if not id_str or not id_str.isdigit():
            return False
        
        num = int(id_str)
        return num >= 1  # Any positive integer
    
    def parse_reference_id(self, reference: str) -> Optional[str]:
        """
        Parse ID reference from command line (e.g., "-123456" -> "123456").
        
        Args:
            reference: Reference string (may include prefix)
            
        Returns:
            str or None: Valid ID or None if invalid
        """
        # Remove common prefixes
        cleaned = reference.strip()
        if cleaned.startswith('-'):
            cleaned = cleaned[1:]
        
        # Validate as positive integer
        if cleaned.isdigit():
            num = int(cleaned)
            if num >= 1:  # Any positive integer
                return str(num)  # Return as-is, no zero padding
        
        return None