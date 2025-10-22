"""
Summarization service for processing text files with AI-powered summaries and tags.
Handles both transcript files and arbitrary text/markdown files.
"""

import os
import shutil
import requests
import json
import yaml
import re
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from file_header import TranscriptHeader


class SummarizationService:
    """Service for AI-powered summarization and tagging of text files."""
    
    def __init__(self, 
                 ollama_model: str,
                 ollama_api_url: str = "http://localhost:11434/api/generate",
                 ollama_timeout: int = 180,
                 notes_folder: Optional[str] = None):
        """
        Initialize the summarization service.
        
        Args:
            ollama_model: Ollama model to use for summarization
            ollama_api_url: Ollama API endpoint
            ollama_timeout: Timeout in seconds for Ollama requests
            notes_folder: Optional folder to copy processed files to
        """
        self.ollama_model = ollama_model
        self.ollama_api_url = ollama_api_url
        self.ollama_timeout = ollama_timeout
        self.notes_folder = notes_folder
        
        # Load prompts for metadata generation
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from prompts.json file."""
        try:
            prompts_path = os.path.join(os.path.dirname(__file__), '..', 'prompts.json')
            with open(prompts_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load prompts file: {e}")
            return {}
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _call_ollama(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call Ollama with retry logic and graceful error handling.
        
        Args:
            prompt: The prompt to send to Ollama
            max_retries: Maximum number of retry attempts
            
        Returns:
            str or None: AI response or None if failed
        """
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 200
                    }
                }
                
                print(f"🤖 Calling Ollama (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(self.ollama_api_url, json=payload, timeout=self.ollama_timeout)
                response.raise_for_status()
                
                # Get response and clean it up
                raw_response = json.loads(response.text)["response"].strip()
                
                # Remove thinking tags and extract JSON
                cleaned = self._clean_ollama_response(raw_response)
                return cleaned if cleaned else None
                
            except requests.exceptions.Timeout:
                timeout_minutes = self.ollama_timeout // 60
                timeout_seconds = self.ollama_timeout % 60
                time_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
                print(f"⚠️ Ollama request timed out after {time_str} (attempt {attempt + 1})")
                
            except requests.exceptions.ConnectionError:
                print(f"⚠️ Could not connect to Ollama (attempt {attempt + 1})")
                
            except Exception as e:
                print(f"⚠️ Ollama error (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                print("   Retrying in 2 seconds...")
                import time
                time.sleep(2)
        
        print(f"❌ Ollama failed after {max_retries} attempts")
        print("💡 Tips:")
        print("   - Check if Ollama is running: ollama list")
        print("   - Try a different model: ollama pull gemma3:4b")
        print("   - Restart Ollama: ollama serve")
        return None
    
    def _clean_ollama_response(self, raw_response: str) -> Optional[str]:
        """Clean up Ollama response and extract JSON."""
        # Remove various thinking patterns
        cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\[thinking\].*?\[/thinking\]', '', cleaned, flags=re.DOTALL)
        
        # Look for JSON blocks with different patterns
        json_patterns = [
            r'\{[^{}]*"filename"[^{}]*"summary"[^{}]*"tags"[^{}]*\}',  # Complete JSON with required fields
            r'\{.*?"filename".*?"summary".*?"tags".*?\}',  # Flexible JSON matching
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'`(\{.*?\})`',  # JSON in backticks
            r'\{.*\}',  # Any JSON-like structure (last resort)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned, re.DOTALL | re.IGNORECASE)
            if matches:
                # Take the first match and clean it up
                json_candidate = matches[0]
                if isinstance(json_candidate, tuple):
                    json_candidate = json_candidate[0] if json_candidate else matches[0]
                
                # Clean up the JSON
                json_candidate = json_candidate.strip()
                if json_candidate:
                    return json_candidate
        
        print(f"⚠️ Could not find valid JSON in response: {raw_response[:200]}...")
        return None
    
    def get_metadata(self, text_content: str) -> Optional[Dict[str, Any]]:
        """
        Get AI-generated metadata (filename, summary, tags) for text content.
        
        Args:
            text_content: Text to analyze
            
        Returns:
            dict or None: Metadata with 'filename', 'summary', 'tags' keys
        """
        if not self.check_ollama_available():
            print("❌ Ollama not available - skipping AI metadata generation")
            return None
        
        if "combined_metadata" not in self.prompts:
            print("❌ Combined metadata prompt not found in prompts.json")
            return None
        
        prompt_template = self.prompts["combined_metadata"]["prompt"]
        
        # Add explicit JSON formatting instruction
        enhanced_prompt = prompt_template + "\n\nIMPORTANT: Return ONLY valid JSON. No explanations, no additional text, just the JSON object."
        prompt = enhanced_prompt.format(text=text_content)
        
        result = self._call_ollama(prompt)
        if not result:
            return None
        
        try:
            # Parse JSON response
            metadata = json.loads(result)
            
            # Validate required fields
            if all(key in metadata for key in ['filename', 'summary', 'tags']):
                # Clean up filename
                filename = metadata['filename'].strip()
                if not filename:
                    return None
                
                # Clean up tags
                tags = metadata['tags']
                if isinstance(tags, list):
                    tags = [tag.strip().lower().replace(' ', '-') for tag in tags if tag.strip()]
                    tags = tags[:5]  # Limit to 5 tags max
                else:
                    tags = []
                
                return {
                    'filename': filename,
                    'summary': metadata['summary'].strip(),
                    'tags': tags
                }
                
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing AI response as JSON: {e}")
            print(f"Raw response: {result[:200]}...")
            return None
        
        return None
    
    def summarize_file(self, file_path: str, copy_to_notes: bool = True) -> bool:
        """
        Summarize and tag a file, updating its frontmatter.
        
        Args:
            file_path: Path to the file to summarize
            copy_to_notes: Whether to copy processed file to notes folder
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract existing frontmatter and content
            has_frontmatter, frontmatter_data, text_content = self._parse_file_content(content)
            
            if not text_content.strip():
                print("❌ No text content found in file")
                return False
            
            print(f"📖 Analyzing file: {os.path.basename(file_path)}")
            print(f"   Content length: {len(text_content)} characters")
            
            # Truncate very long content to avoid Ollama timeouts
            if len(text_content) > 8000:
                print(f"   ⚠️ Content is very long, using first 8000 characters for analysis")
                text_content = text_content[:8000] + "..."
            
            # Get AI metadata
            metadata = self.get_metadata(text_content)
            if not metadata:
                print("⚠️ Could not generate AI metadata - file not processed")
                return False
            
            # Update or create frontmatter
            updated_content = self._update_frontmatter(
                content, has_frontmatter, frontmatter_data, metadata, text_content
            )
            
            # Determine output path
            output_path = file_path
            
            # If no frontmatter existed and notes folder specified, copy to notes folder
            if not has_frontmatter and self.notes_folder and copy_to_notes:
                os.makedirs(self.notes_folder, exist_ok=True)
                filename = metadata['filename']
                # Clean filename for filesystem
                clean_filename = self._clean_filename(filename)
                date_str = datetime.now().strftime("%d%m%Y")
                
                # Determine extension
                _, ext = os.path.splitext(file_path)
                if not ext:
                    ext = '.md'
                
                new_filename = f"{clean_filename}_{date_str}{ext}"
                output_path = os.path.join(self.notes_folder, new_filename)
                
                print(f"📋 Creating new note: {new_filename}")
            else:
                print(f"📝 Updating existing file: {os.path.basename(file_path)}")
            
            # Write updated content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✅ File processed successfully!")
            print(f"   📁 Saved to: {output_path}")
            print(f"   📝 Summary: {metadata['summary']}")
            if metadata['tags']:
                print(f"   🏷️  Tags: {', '.join(metadata['tags'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return False
    
    def _parse_file_content(self, content: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Parse file content to separate frontmatter from text content.
        
        Returns:
            tuple: (has_frontmatter, frontmatter_data, text_content)
        """
        # Try to parse existing YAML frontmatter
        yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.MULTILINE | re.DOTALL)
        match = yaml_pattern.match(content)
        
        if match:
            try:
                yaml_content = match.group(1)
                frontmatter_data = yaml.safe_load(yaml_content)
                
                # Remove frontmatter from content
                content_after_frontmatter = yaml_pattern.sub('', content, count=1)
                
                # For transcript files, extract just the transcription section
                transcription_match = re.search(r'## 🎙️ Transcription\s*\n\s*(.*)', content_after_frontmatter, re.DOTALL)
                if transcription_match:
                    text_content = transcription_match.group(1).strip()
                else:
                    # Remove other markdown sections and headers
                    text_content = re.sub(r'^## 📝 Summary.*?^---\s*$', '', content_after_frontmatter, flags=re.MULTILINE | re.DOTALL)
                    text_content = re.sub(r'^## .*?\n', '', text_content, flags=re.MULTILINE)
                    text_content = text_content.strip()
                
                return True, frontmatter_data, text_content
            except yaml.YAMLError:
                pass
        
        # No valid frontmatter found
        return False, None, content.strip()
    
    def _update_frontmatter(self, original_content: str, has_frontmatter: bool, 
                          frontmatter_data: Optional[Dict], metadata: Dict[str, Any], 
                          text_content: str) -> str:
        """Update or create frontmatter with new metadata."""
        
        if has_frontmatter and frontmatter_data:
            # Update existing frontmatter
            updated_frontmatter = frontmatter_data.copy()
            updated_frontmatter['summary'] = metadata['summary']
            updated_frontmatter['tags'] = metadata['tags']
            
            # Add processing timestamp
            updated_frontmatter['last_processed'] = datetime.now().isoformat()
            
            # Generate new YAML
            yaml_content = yaml.dump(updated_frontmatter, default_flow_style=False, sort_keys=False)
            
            # Replace original frontmatter and update summary section
            yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.MULTILINE | re.DOTALL)
            
            # First, replace the frontmatter
            content_after_frontmatter = yaml_pattern.sub('', original_content, count=1)
            
            # Update the summary section if it exists
            summary_pattern = re.compile(r'(## 📝 Summary\s*\n\s*>\s*)\[Summary will be generated if requested\]', re.MULTILINE)
            if summary_pattern.search(content_after_frontmatter):
                content_after_frontmatter = summary_pattern.sub(f'\\1{metadata["summary"]}', content_after_frontmatter)
            
            # Combine new frontmatter with updated content
            return f"---\n{yaml_content}---\n{content_after_frontmatter}"
            
        else:
            # Create new frontmatter for files without existing structure
            new_frontmatter = {
                'title': metadata['filename'],
                'summary': metadata['summary'],
                'tags': metadata['tags'],
                'created': datetime.now().isoformat(),
                'status': 'processed'
            }
            
            yaml_content = yaml.dump(new_frontmatter, default_flow_style=False, sort_keys=False)
            
            # Create new file with frontmatter and content
            return f"---\n{yaml_content}---\n\n## 📝 Summary\n\n> {metadata['summary']}\n\n---\n\n## 📄 Content\n\n{text_content}\n"
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename to make it filesystem-safe."""
        import string
        
        # Remove or replace invalid characters
        valid_chars = f"-_.{string.ascii_letters}{string.digits}"
        cleaned = ''.join(c if c in valid_chars else '_' for c in filename)
        
        # Remove multiple underscores and trim
        cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        
        # Ensure it's not empty and not too long
        if not cleaned:
            cleaned = "processed_file"
        elif len(cleaned) > 50:
            cleaned = cleaned[:50].strip('_')
        
        return cleaned