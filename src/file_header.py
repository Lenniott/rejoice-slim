"""
File header management for transcript files.
Handles YAML front matter with ID and metadata.
"""

import os
import yaml
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import copy


class TranscriptHeader:
    """Manages YAML front matter for transcript files."""
    
    # Class-level cache for template configuration
    _template_cache: Optional[Dict] = None
    
    def __init__(self, transcript_id: str, creation_date: Optional[datetime] = None):
        """
        Initialize transcript header.
        
        Args:
            transcript_id: 6-digit transcript ID
            creation_date: Creation timestamp (defaults to now)
        """
        self.transcript_id = transcript_id
        self.creation_date = creation_date or datetime.now()
    
    @classmethod
    def _load_template_config(cls) -> Dict:
        """
        Load template configuration from template_config.yaml.
        Uses class-level caching to avoid repeated file reads.
        
        Returns:
            dict: Template configuration
        """
        if cls._template_cache is not None:
            return cls._template_cache
        
        # Get path to template config (in project root)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        template_path = os.path.join(project_root, 'template_config.yaml')
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                cls._template_cache = config
                return config
        except (IOError, OSError, yaml.YAMLError) as e:
            # Fallback to default template if config file not found
            print(f"Warning: Could not load template_config.yaml: {e}")
            print("Using default template structure.")
            cls._template_cache = {
                'frontmatter_template': {
                    'area': '',
                    'category': '',
                    'id': '{{id}}',
                    'type': '',
                    'status': 'draft',
                    'linked': [],
                    'tags': [],
                    'created': '{{created}}',
                    'archive_by': '{{archive_by}}'
                }
            }
            return cls._template_cache
    
    def _replace_placeholders(self, template_data: Dict) -> Dict:
        """
        Replace template placeholders with actual values.
        
        Available placeholders:
        - {{id}}         : 6-digit transcript ID
        - {{created}}    : ISO format creation timestamp (YYYY-MM-DD HH:mm)
        - {{archive_by}} : Date 30 days from creation (YYYY-MM-DD)
        
        Args:
            template_data: Template dictionary with placeholders
            
        Returns:
            dict: Template with placeholders replaced
        """
        # Create a deep copy to avoid modifying the cached template
        data = copy.deepcopy(template_data)
        
        # Calculate values for placeholders
        created_str = self.creation_date.strftime('%Y-%m-%d %H:%M')
        archive_by_date = self.creation_date + timedelta(days=30)
        archive_by_str = archive_by_date.strftime('%Y-%m-%d')
        
        # Replacement map
        replacements = {
            '{{id}}': self.transcript_id,
            '{{created}}': created_str,
            '{{archive_by}}': archive_by_str
        }
        
        # Recursively replace placeholders in the dictionary
        def replace_in_value(value):
            if isinstance(value, str):
                for placeholder, replacement in replacements.items():
                    value = value.replace(placeholder, replacement)
                return value
            elif isinstance(value, dict):
                return {k: replace_in_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_in_value(item) for item in value]
            return value
        
        return replace_in_value(data)
    
    def generate_header_yaml(self) -> str:
        """
        Generate YAML front matter header using template from template_config.yaml.
        
        The template supports placeholders that are automatically replaced:
        - {{id}}         : 6-digit transcript ID
        - {{created}}    : ISO format creation timestamp (YYYY-MM-DD HH:mm)
        - {{archive_by}} : Date 30 days from creation (YYYY-MM-DD)
        
        Returns:
            str: YAML front matter as string
        """
        # Load template configuration
        config = self._load_template_config()
        template = config.get('frontmatter_template', {})
        
        # Replace placeholders with actual values
        header_data = self._replace_placeholders(template)
        
        # Generate YAML with proper formatting
        yaml_content = yaml.dump(header_data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        return f"---\n{yaml_content}---\n\n"
    
    def create_file_content(self, transcript_text: str, output_format: str = "md") -> str:
        """
        Create complete file content with header and transcript.

        Args:
            transcript_text: Transcribed text content
            output_format: Output format ("md" or "txt")

        Returns:
            str: Complete file content
        """
        header = self.generate_header_yaml()

        if output_format == "md":
            return f"{header}## 🎙️ Transcription\n\n```\n{transcript_text}\n```\n"
        else:  # txt format
            return f"{header}Transcription:\n{transcript_text}\n"
    
    @staticmethod
    def parse_header(file_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse YAML front matter from file content.
        
        Args:
            file_content: Complete file content
            
        Returns:
            dict or None: Parsed header data or None if no valid header
        """
        # Match YAML front matter pattern
        yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.MULTILINE | re.DOTALL)
        match = yaml_pattern.match(file_content)
        
        if not match:
            return None
        
        try:
            yaml_content = match.group(1)
            header_data = yaml.safe_load(yaml_content)
            return header_data
        except yaml.YAMLError:
            return None
    
    @staticmethod
    def extract_transcript_content(file_content: str) -> str:
        """
        Extract just the transcript content, removing header.
        
        Args:
            file_content: Complete file content
            
        Returns:
            str: Transcript content without header
        """
        # Remove YAML front matter
        yaml_pattern = re.compile(r'^---\s*\n.*?\n---\s*\n', re.MULTILINE | re.DOTALL)
        content_without_header = yaml_pattern.sub('', file_content)
        
        # Remove markdown section headers if present
        content_without_header = re.sub(r'^## 🎙️ Transcription\s*\n', '', content_without_header, flags=re.MULTILINE)
        content_without_header = re.sub(r'^Transcription:\s*\n', '', content_without_header, flags=re.MULTILINE)
        
        return content_without_header.strip()
    
    @staticmethod
    def get_id_from_file(filepath: str) -> Optional[str]:
        """
        Extract transcript ID from file content or filename.
        
        Args:
            filepath: Path to transcript file
            
        Returns:
            str or None: Transcript ID or None if not found
        """
        # First try to extract from filename
        filename = os.path.basename(filepath)
        pattern = re.compile(r'^(\d+)_\d{8}_.*\.(txt|md)$')
        match = pattern.match(filename)
        if match:
            return match.group(1)
        
        # Fallback to reading from file content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            header_data = TranscriptHeader.parse_header(content)
            if header_data and 'id' in header_data:
                return str(header_data['id'])
        except (IOError, OSError):
            pass
        
        return None
    
    @staticmethod
    def is_id_format_file(filename: str) -> bool:
        """
        Check if filename follows the new ID format.
        Format: {id}_DDMMYYYY_{generated_name}.{ext}
        
        Args:
            filename: Filename to check
            
        Returns:
            bool: True if filename matches new ID format
        """
        id_pattern = re.compile(r'^\d+_\d{8}_.+\.(txt|md)$')
        return bool(id_pattern.match(filename))
    
    @staticmethod
    def is_legacy_format_file(filename: str) -> bool:
        """
        Check if filename follows the legacy timestamp format.
        
        Args:
            filename: Filename to check
            
        Returns:
            bool: True if filename appears to be legacy format
        """
        # Legacy format: transcript-20251015-114500.txt or similar timestamp patterns
        legacy_patterns = [
            r'^transcript-\d{8}-\d{6}\.(txt|md)$',  # transcript-YYYYMMDD-HHMMSS
            r'^\d{8}_\d{4}_[a-f0-9]{6}\.(txt|md)$',  # YYYYMMDD_HHMM_uuid
        ]
        
        for pattern in legacy_patterns:
            if re.match(pattern, filename):
                return True
        
        return False