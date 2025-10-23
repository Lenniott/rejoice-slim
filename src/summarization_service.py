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
from typing import Optional, Dict, Any, Tuple, List
from file_header import TranscriptHeader


class SummarizationService:
    """Service for AI-powered summarization and tagging of text files."""
    
    def __init__(self, 
                 ollama_model: str,
                 ollama_api_url: str = "http://localhost:11434/api/generate",
                 ollama_timeout: int = 180,
                 notes_folder: Optional[str] = None,
                 max_content_length: int = 32000):
        """
        Initialize the summarization service.
        
        Args:
            ollama_model: Ollama model to use for summarization
            ollama_api_url: Ollama API endpoint
            ollama_timeout: Timeout in seconds for Ollama requests
            notes_folder: Optional folder to copy processed files to
            max_content_length: Maximum characters to send to AI (default: 32000)
        """
        self.ollama_model = ollama_model
        self.ollama_api_url = ollama_api_url
        self.ollama_timeout = ollama_timeout
        self.notes_folder = notes_folder
        self.max_content_length = max_content_length
        
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
                    "format": "json",
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 400
                    }
                }
                
                response = requests.post(self.ollama_api_url, json=payload, timeout=self.ollama_timeout)
                response.raise_for_status()
                
                # Get response and clean it up
                response_data = json.loads(response.text)
                raw_response = response_data.get("response", "").strip()
                
                if not raw_response:
                    continue
                
                # With structured outputs, response should already be valid JSON
                if raw_response.strip():
                    return raw_response.strip()
                else:
                    continue
                
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:  # Only show on last attempt
                    timeout_minutes = self.ollama_timeout // 60
                    timeout_seconds = self.ollama_timeout % 60
                    time_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
                    print(f"⚠️ Ollama timeout after {time_str}")
                
            except requests.exceptions.ConnectionError:
                if attempt == max_retries - 1:  # Only show on last attempt
                    print(f"⚠️ Could not connect to Ollama")
                
            except Exception as e:
                if attempt == max_retries - 1:  # Only show on last attempt
                    print(f"⚠️ Ollama error: {e}")
            
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
        
        print(f"❌ Ollama failed after {max_retries} attempts")
        return None
    
    def _sanitize_json_string(self, text: str) -> str:
        """Sanitize text for safe JSON string usage."""
        # Replace problematic characters that break JSON
        text = text.replace('\\', '\\\\')  # Escape backslashes first
        text = text.replace('"', '\\"')    # Escape quotes
        text = text.replace('\n', ' ')     # Replace newlines with spaces
        text = text.replace('\r', ' ')     # Replace carriage returns
        text = text.replace('\t', ' ')     # Replace tabs
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _repair_json(self, json_str: str) -> Optional[str]:
        """Attempt to repair common JSON syntax issues."""
        try:
            # First try parsing as-is
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            print(f"   🔧 Attempting JSON repair for error: {e}")
            
            # Start with original
            repaired = json_str
            
            # More aggressive quote fixing - handle unescaped quotes in string values
            # Pattern: "field": "text with "embedded quotes" here"
            def fix_field_quotes(field_name):
                # Find the field and extract its value, fixing embedded quotes
                pattern = rf'"{field_name}":\s*"(.*?)"(?=\s*[,}}])'
                
                def quote_fixer(match):
                    value = match.group(1)
                    # Count quotes to see if we have unbalanced ones
                    quote_count = value.count('"')
                    if quote_count > 0:
                        # Replace all internal quotes with escaped quotes or remove them
                        # Try escaping first
                        fixed_value = value.replace('"', '\\"')
                        return f'"{field_name}": "{fixed_value}"'
                    return match.group(0)
                
                return re.sub(pattern, quote_fixer, repaired, flags=re.DOTALL)
            
            # Apply to all known string fields
            for field in ['filename', 'summary']:
                repaired = fix_field_quotes(field)
            
            # If that doesn't work, try more aggressive cleaning
            if repaired == json_str:
                # Remove all internal quotes from string values
                def aggressive_quote_removal(match):
                    field = match.group(1)
                    value = match.group(2)
                    # Remove all quotes and problematic characters
                    cleaned_value = value.replace('"', '').replace('\n', ' ').replace('\r', ' ')
                    cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
                    return f'"{field}": "{cleaned_value}"'
                
                # Apply to string fields
                repaired = re.sub(r'"(filename|summary)":\s*"([^"]*(?:"[^"]*)*)"', aggressive_quote_removal, repaired)
            
            # Fix trailing commas
            repaired = re.sub(r',\s*}', '}', repaired)
            repaired = re.sub(r',\s*]', ']', repaired)
            
            # Fix newlines in strings
            repaired = re.sub(r'"([^"]*)\n([^"]*)"', r'"\1 \2"', repaired)
            
            # Try parsing the repaired version
            try:
                json.loads(repaired)
                print(f"   ✅ JSON repair successful")
                return repaired
            except json.JSONDecodeError as repair_error:
                print(f"   ❌ JSON repair failed: {repair_error}")
                print(f"   🔧 Repaired JSON was: {repaired[:200]}...")
                return None

    def _clean_ollama_response(self, raw_response: str) -> Optional[str]:
        """Clean up Ollama response and extract JSON."""
        # Reduced verbose output - only show on debug if needed
        
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
        
        for i, pattern in enumerate(json_patterns):
            matches = re.findall(pattern, cleaned, re.DOTALL | re.IGNORECASE)
            if matches:
                # Take the first match and clean it up
                json_candidate = matches[0]
                if isinstance(json_candidate, tuple):
                    json_candidate = json_candidate[0] if json_candidate else matches[0]
                
                # Clean up the JSON
                json_candidate = json_candidate.strip()
                if json_candidate:
                    # Try to repair common JSON issues
                    repaired = self._repair_json(json_candidate)
                    if repaired:
                        return repaired
                    return json_candidate
        
        # Could not extract valid JSON from response
        return None
    
    def get_metadata(self, text_content: str) -> Optional[Dict[str, Any]]:
        """
        Get AI-generated metadata (filename, summary, tags) for text content.
        Uses hierarchical summarization for large content.
        
        Args:
            text_content: Text to analyze
            
        Returns:
            dict or None: Metadata with 'filename', 'summary', 'tags' keys
        """
        if not self.check_ollama_available():
            return None
        
        # Use hierarchical summarization for large content
        if len(text_content) > 3000:
            return self._hierarchical_summarize(text_content)
        
        # Use single-step for smaller content
        if "combined_metadata" not in self.prompts:
            return None
        
        # Handle very large content by truncating if necessary
        max_content_length = int(os.getenv('OLLAMA_MAX_CONTENT_LENGTH', '15000'))
        if len(text_content) > max_content_length:
            # Take first 80% and last 20% to preserve more beginning context where topics are mentioned
            split_point = int(max_content_length * 0.8)
            remaining = max_content_length - split_point
            truncated_content = text_content[:split_point] + "\n\n[... content truncated ...]\n\n" + text_content[-remaining:]
            text_content = truncated_content
        
        prompt_template = self.prompts["combined_metadata"]["prompt"]
        prompt = prompt_template.format(text=text_content)
        
        result = self._call_ollama(prompt)
        if not result:
            return None
        
        # If we get a result, try to parse it first
        parsed_result = self._try_parse_metadata(result)
        if parsed_result:
            return parsed_result
        
        # If parsing failed, try a simpler fallback prompt with better content sampling
        
        # Take a better sample: beginning + middle + end
        content_len = len(text_content)
        if content_len > 2000:
            sample = text_content[:600] + "\n[...middle content...]\n" + text_content[content_len//2:content_len//2+400] + "\n[...end content...]\n" + text_content[-600:]
        else:
            sample = text_content
            
        fallback_prompt = f"""Analyze this content and respond ONLY with valid JSON. No extra text.

Content sample:
{sample}

Required format:
{{"filename": "descriptive-topic-name", "summary": "Brief 2-3 sentence description", "tags": ["keyword1", "keyword2", "keyword3"]}}

JSON:"""
        
        fallback_result = self._call_ollama(fallback_prompt)
        if fallback_result:
            return self._try_parse_metadata(fallback_result)
        
        return None

    def _create_smart_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Create chunks with smart boundary detection to avoid cutting mid-sentence.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:].strip())
                break
            
            # Look for good break points near the target end
            # Priority: paragraph break > sentence end > word boundary
            break_point = end
            
            # Look backward from end for good break points (up to 200 chars)
            search_start = max(start + chunk_size - 200, start)
            search_text = text[search_start:end + 100]  # Look a bit ahead too
            
            # Find paragraph breaks (double newlines)
            paragraph_breaks = [m.start() + search_start for m in re.finditer(r'\n\s*\n', search_text)]
            if paragraph_breaks:
                suitable_breaks = [bp for bp in paragraph_breaks if bp > start + chunk_size // 2]
                if suitable_breaks:
                    break_point = suitable_breaks[0]
            
            # If no paragraph break, find sentence endings
            if break_point == end:
                sentence_endings = [m.end() + search_start for m in re.finditer(r'[.!?]\s+', search_text)]
                suitable_endings = [se for se in sentence_endings if start + chunk_size // 2 < se < start + chunk_size + 100]
                if suitable_endings:
                    break_point = suitable_endings[-1]  # Take the last suitable sentence end
            
            # If still no good break, find word boundaries
            if break_point == end:
                word_boundaries = [m.start() + search_start for m in re.finditer(r'\s+', search_text)]
                suitable_words = [wb for wb in word_boundaries if start + chunk_size // 2 < wb < start + chunk_size + 50]
                if suitable_words:
                    break_point = suitable_words[-1]
            
            # Create chunk
            chunk = text[start:break_point].strip()
            if chunk:
                chunks.append(chunk)
            
            # Next chunk starts with overlap
            start = max(break_point - overlap, start + 1)
        
        return chunks

    def _hierarchical_summarize(self, text_content: str) -> Optional[Dict[str, Any]]:
        """
        Hierarchical summarization: chunk -> summarize each -> meta-summarize.
        Focuses on questions, actions, and narrative threads.
        """
        if "chunk_summary" not in self.prompts or "meta_summary" not in self.prompts:
            print("❌ Hierarchical summarization prompts not found")
            return None
        
        # Step 1: Break into chunks with smart boundary detection
        chunk_size = 1800   # Slightly smaller for more consistent processing
        overlap = 150       # Reduced overlap for cleaner boundaries
        chunks = self._create_smart_chunks(text_content, chunk_size, overlap)
        
        # Step 2: Summarize each chunk
        chunk_summaries = []
        chunk_prompt_template = self.prompts["chunk_summary"]["prompt"]
        
        for i, chunk in enumerate(chunks):
            chunk_prompt = chunk_prompt_template.format(text=chunk)
            
            chunk_result = self._call_ollama(chunk_prompt)
            if chunk_result:
                # Clean the result and extract just the summary text
                summary = chunk_result.strip()
                # Remove any JSON formatting that might have crept in
                summary = re.sub(r'^[{\'"]*', '', summary)
                summary = re.sub(r'[}\'"]*$', '', summary) 
                chunk_summaries.append(f"Chunk {i+1}: {summary}")
            else:
                print(f"   ⚠️ Failed to summarize chunk {i+1}")
        
        if not chunk_summaries:
            print("❌ No chunks were successfully summarized")
            return None
        
        # Step 3: Create meta-summary from chunk summaries
        combined_summaries = "\n".join(chunk_summaries)
        meta_prompt_template = self.prompts["meta_summary"]["prompt"]
        meta_prompt = meta_prompt_template.format(text=combined_summaries)
        meta_result = self._call_ollama(meta_prompt)
        if not meta_result:
            print("❌ Failed to create meta-summary")
            return None
        
        # Step 4: Parse the final result
        return self._try_parse_metadata(meta_result)

    def _try_parse_metadata(self, result: str) -> Optional[Dict[str, Any]]:
        
        try:
            # Parse JSON response
            metadata = json.loads(result)
            
            # Validate required fields
            required_fields = ['filename', 'summary', 'tags']
            
            if all(key in metadata for key in required_fields):
                # Clean up filename
                filename = metadata['filename'].strip()
                if not filename:
                    return None
                
                # Validate and clean up summary
                summary = metadata['summary'].strip()
                if not summary:
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
                    'summary': summary,
                    'tags': tags
                }
            else:
                return None
                
        except json.JSONDecodeError as e:
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
            
            # Truncate very long content to avoid Ollama timeouts
            if len(text_content) > self.max_content_length:
                text_content = text_content[:self.max_content_length] + "..."
            
            # Get AI metadata
            metadata = self.get_metadata(text_content)
            if not metadata:
                return False
            
            # Update or create frontmatter
            updated_content = self._update_frontmatter(
                content, has_frontmatter, frontmatter_data, metadata, text_content
            )
            
            # Determine output path and handle renaming
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
                # Check if this is a transcript file that can be renamed with AI-generated filename
                if has_frontmatter and self._is_transcript_file(file_path, frontmatter_data):
                    new_output_path = self._rename_transcript_with_ai_filename(file_path, metadata['filename'], frontmatter_data)
                    if new_output_path and new_output_path != file_path:
                        output_path = new_output_path
            
            # Write updated content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
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
    
    def _is_transcript_file(self, file_path: str, frontmatter_data: Dict) -> bool:
        """
        Check if this is a transcript file that can be renamed.
        
        Args:
            file_path: Path to the file
            frontmatter_data: Parsed frontmatter data
            
        Returns:
            bool: True if this is a renameable transcript file
        """
        # Check if it has transcript-like frontmatter (id field indicates transcript)
        if 'id' in frontmatter_data:
            return True
            
        # Check if filename matches transcript pattern: *_DDMMYYYY_ID.ext
        filename = os.path.basename(file_path)
        transcript_pattern = re.compile(r'^.*_\d{8}_\d+\.(md|txt)$')
        return bool(transcript_pattern.match(filename))
    
    def _rename_transcript_with_ai_filename(self, file_path: str, ai_filename: str, frontmatter_data: Dict) -> Optional[str]:
        """
        Rename transcript file with AI-generated filename while preserving ID structure.
        
        Args:
            file_path: Current file path
            ai_filename: AI-generated filename
            frontmatter_data: Parsed frontmatter data
            
        Returns:
            str or None: New file path if renamed successfully, None otherwise
        """
        try:
            directory = os.path.dirname(file_path)
            current_filename = os.path.basename(file_path)
            
            # Extract ID and date from current filename
            # Pattern: *_DDMMYYYY_ID.ext
            pattern = re.compile(r'^.*_(\d{8})_(\d+)\.(.+)$')
            match = pattern.match(current_filename)
            
            if not match:
                # Try to get ID from frontmatter
                if 'id' in frontmatter_data:
                    # Use current date if no date pattern found
                    date_str = datetime.now().strftime("%d%m%Y")
                    file_id = str(frontmatter_data['id'])
                    ext = os.path.splitext(file_path)[1][1:]  # Remove the dot
                else:
                    return None
            else:
                date_str = match.group(1)
                file_id = match.group(2)  
                ext = match.group(3)
            
            # Create new filename with AI-generated name
            clean_ai_filename = self._clean_filename(ai_filename)
            new_filename = f"{clean_ai_filename}_{date_str}_{file_id}.{ext}"
            new_file_path = os.path.join(directory, new_filename)
            
            # Only rename if the new name is different
            if new_file_path != file_path:
                os.rename(file_path, new_file_path)
                return new_file_path
                
            return file_path
            
        except Exception as e:
            print(f"   ⚠️ Could not rename file: {e}")
            return None