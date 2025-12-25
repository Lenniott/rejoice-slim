"""
Text formatting service for reformatting text into meaningful paragraphs using Ollama.
Handles chunked processing with overlap and supports thinking models.
"""

import os
import re
import json
import requests
from typing import Optional, List
from dotenv import load_dotenv


class FormattingService:
    """Service for AI-powered text formatting into meaningful paragraphs."""

    def __init__(self,
                 ollama_model: str,
                 ollama_api_url: str = "http://localhost:11434/api/generate",
                 ollama_timeout: int = 180,
                 chunk_size: int = 1000,
                 overlap_size: int = 200):
        """
        Initialize the formatting service.

        Args:
            ollama_model: Ollama model to use for formatting
            ollama_api_url: Ollama API endpoint (must be localhost for security)
            ollama_timeout: Timeout in seconds for Ollama requests
            chunk_size: Size of text chunks to process (default: 1000 chars)
            overlap_size: Overlap between chunks (default: 200 chars)

        Raises:
            ValueError: If ollama_api_url points to a non-localhost address
        """
        # Security: Validate that Ollama URL is localhost-only
        self._validate_localhost_url(ollama_api_url)

        self.ollama_model = ollama_model
        self.ollama_api_url = ollama_api_url
        self.ollama_timeout = ollama_timeout
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def _validate_localhost_url(self, url: str) -> None:
        """
        Validate that the Ollama API URL points to localhost only.

        This is a critical security check to ensure text is never
        sent to remote servers, maintaining privacy.

        Args:
            url: The Ollama API URL to validate

        Raises:
            ValueError: If URL points to a non-localhost address
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            hostname = parsed.hostname

            # List of valid localhost identifiers
            valid_hosts = ['localhost', '127.0.0.1', '::1', '0.0.0.0']

            if hostname is None:
                raise ValueError(
                    f"⚠️  SECURITY ERROR: Invalid Ollama API URL format: {url}\n"
                    "   URL must explicitly specify localhost (e.g., http://localhost:11434/api/generate)"
                )

            if hostname.lower() not in valid_hosts:
                raise ValueError(
                    f"⚠️  SECURITY ERROR: Ollama API URL must be localhost, got: {hostname}\n"
                    f"   Attempted URL: {url}\n\n"
                    "   Only local Ollama instances are supported to ensure your text\n"
                    "   remains private and is never sent to remote servers.\n\n"
                    "   Valid examples:\n"
                    "     • http://localhost:11434/api/generate\n"
                    "     • http://127.0.0.1:11434/api/generate\n"
                )

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"⚠️  SECURITY ERROR: Could not parse Ollama API URL: {url}\n"
                f"   Error: {e}\n"
                "   Please use a valid localhost URL (e.g., http://localhost:11434/api/generate)"
            )

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
        Handles thinking models by extracting content after thinking tags.

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
                        "top_p": 0.9
                    }
                }

                response = requests.post(self.ollama_api_url, json=payload, timeout=self.ollama_timeout)
                response.raise_for_status()

                # Get response
                response_data = json.loads(response.text)
                raw_response = response_data.get("response", "").strip()

                if not raw_response:
                    continue

                # Handle thinking models - remove thinking tags
                cleaned_response = self._remove_thinking_tags(raw_response)

                if cleaned_response.strip():
                    return cleaned_response.strip()
                else:
                    continue

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    timeout_minutes = self.ollama_timeout // 60
                    timeout_seconds = self.ollama_timeout % 60
                    time_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
                    print(f"⚠️ Ollama timeout after {time_str}")

            except requests.exceptions.ConnectionError:
                if attempt == max_retries - 1:
                    print(f"⚠️ Could not connect to Ollama")

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⚠️ Ollama error: {e}")

            if attempt < max_retries - 1:
                import time
                time.sleep(2)

        print(f"❌ Ollama failed after {max_retries} attempts")
        return None

    def _remove_thinking_tags(self, text: str) -> str:
        """
        Remove thinking tags from model output.
        Supports various thinking tag formats: <think>, <thinking>, [thinking]

        Args:
            text: Raw model output

        Returns:
            str: Text with thinking tags removed
        """
        # Remove various thinking patterns
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\[thinking\].*?\[/thinking\]', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        return cleaned.strip()

    def _create_chunks_with_overlap(self, text: str) -> List[str]:
        """
        Create chunks with specified overlap, respecting paragraph boundaries.

        Args:
            text: Text to chunk

        Returns:
            List[str]: List of text chunks with overlap
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # Calculate end position
            end = start + self.chunk_size

            if end >= text_len:
                # Last chunk
                chunks.append(text[start:].strip())
                break

            # Look for good break points near the target end
            # Priority: paragraph break > sentence end > word boundary
            break_point = end

            # Look backward from end for good break points (up to 200 chars)
            search_start = max(start + self.chunk_size - 200, start)
            search_text = text[search_start:end + 100]  # Look a bit ahead too

            # Find paragraph breaks (double newlines)
            paragraph_breaks = [m.start() + search_start for m in re.finditer(r'\n\s*\n', search_text)]
            if paragraph_breaks:
                suitable_breaks = [bp for bp in paragraph_breaks if bp > start + self.chunk_size // 2]
                if suitable_breaks:
                    break_point = suitable_breaks[0]

            # If no paragraph break, find sentence endings
            if break_point == end:
                sentence_endings = [m.end() + search_start for m in re.finditer(r'[.!?]\s+', search_text)]
                suitable_endings = [se for se in sentence_endings if start + self.chunk_size // 2 < se < start + self.chunk_size + 100]
                if suitable_endings:
                    break_point = suitable_endings[-1]

            # If still no good break, find word boundaries
            if break_point == end:
                word_boundaries = [m.start() + search_start for m in re.finditer(r'\s+', search_text)]
                suitable_words = [wb for wb in word_boundaries if start + self.chunk_size // 2 < wb < start + self.chunk_size + 50]
                if suitable_words:
                    break_point = suitable_words[-1]

            # Create chunk
            chunk = text[start:break_point].strip()
            if chunk:
                chunks.append(chunk)

            # Next chunk starts with overlap
            start = max(break_point - self.overlap_size, start + 1)

        return chunks

    def format_text(self, text: str, show_progress: bool = True, total_chunks: int = 0) -> Optional[str]:
        """
        Format text into meaningful paragraphs using Ollama.

        Args:
            text: Text to format
            show_progress: Whether to show progress during processing
            total_chunks: Total number of chunks (for progress display)

        Returns:
            str or None: Formatted text or None if failed
        """
        if not self.check_ollama_available():
            print("❌ Ollama is not available. Please start Ollama first.")
            return None

        # Create chunks with overlap
        chunks = self._create_chunks_with_overlap(text)

        # Format each chunk
        formatted_chunks = []
        prompt_template = """Format the following text into clear, meaningful paragraphs.

Rules:
- Organize related sentences into coherent paragraphs
- Add appropriate paragraph breaks for clarity
- Preserve the original content and meaning
- Do not add, remove, or summarize content
- Return ONLY the formatted text, nothing else

Text to format:

{text}

Formatted text:"""

        for i, chunk in enumerate(chunks):
            if show_progress:
                # Calculate progress percentage
                progress = int((i / len(chunks)) * 100)
                self._show_formatting_progress(i + 1, len(chunks), progress)

            prompt = prompt_template.format(text=chunk)
            formatted_chunk = self._call_ollama(prompt)

            if formatted_chunk:
                formatted_chunks.append(formatted_chunk)
            else:
                if show_progress:
                    print(f"\n⚠️ Failed to format chunk {i+1}, using original")
                formatted_chunks.append(chunk)

        # Combine chunks, removing overlap duplicates
        return self._merge_chunks(formatted_chunks)

    def _show_formatting_progress(self, current_chunk: int, total_chunks: int, progress: int):
        """Show formatting progress in consistent UI style."""
        import os

        # Clear screen
        os.system('clear' if os.name != 'nt' else 'cls')

        print("___________________________________\n")
        print("FORMATTING TEXT...\n")

        # Progress bar
        filled = int(progress * 30 / 100)
        progress_bar = "█" * filled + "░" * (30 - filled)
        print(f"PROGRESS      {progress_bar} {progress}%")
        print(f"CHUNK         {current_chunk}/{total_chunks}")
        print("___________________________________")

    def _merge_chunks(self, chunks: List[str]) -> str:
        """
        Merge formatted chunks, attempting to remove duplicate overlap content.
        Uses sentence-based matching for better overlap detection.

        Args:
            chunks: List of formatted chunks

        Returns:
            str: Merged text
        """
        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0]

        merged = chunks[0]

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]

            # Split both into sentences for better matching
            merged_sentences = self._split_into_sentences(merged)
            current_sentences = self._split_into_sentences(current_chunk)

            if not merged_sentences or not current_sentences:
                merged += "\n\n" + current_chunk
                continue

            # Find overlapping sentences
            # Look at the last few sentences of merged
            max_check = min(10, len(merged_sentences))
            overlap_found = False

            for check_count in range(max_check, 0, -1):
                # Get last N sentences from merged
                check_sentences = merged_sentences[-check_count:]
                check_text = " ".join(check_sentences).strip()

                # See if these appear at the start of current
                current_start = " ".join(current_sentences[:check_count]).strip()

                # Fuzzy match - normalize whitespace and compare
                check_normalized = re.sub(r'\s+', ' ', check_text.lower())
                current_normalized = re.sub(r'\s+', ' ', current_start.lower())

                if check_normalized == current_normalized:
                    # Found overlap - skip these sentences in current chunk
                    remaining_sentences = current_sentences[check_count:]
                    if remaining_sentences:
                        merged += "\n\n" + " ".join(remaining_sentences)
                    overlap_found = True
                    break

            if not overlap_found:
                # No sentence overlap found, just append
                merged += "\n\n" + current_chunk

        return merged.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using basic heuristics.

        Args:
            text: Text to split

        Returns:
            List[str]: List of sentences
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_transcript_content(self, content: str) -> tuple[str, str, str]:
        """
        Extract the transcription section from a transcript file.

        Returns:
            tuple: (prefix, transcript_text, suffix)
                   prefix: Everything before the transcription
                   transcript_text: The actual transcription content
                   suffix: Everything after the transcription
        """
        # Try to find transcription section with ## 🎙️ Transcription header
        transcription_pattern = re.compile(
            r'(.*?## 🎙️ Transcription\s*\n\s*```\s*\n)(.*?)(\n```.*)',
            re.DOTALL
        )
        match = transcription_pattern.search(content)

        if match:
            return match.group(1), match.group(2).strip(), match.group(3)

        # Try to find content between triple backticks (without header)
        code_block_pattern = re.compile(
            r'(.*?```\s*\n)(.*?)(\n```.*)',
            re.DOTALL
        )
        match = code_block_pattern.search(content)

        if match:
            return match.group(1), match.group(2).strip(), match.group(3)

        # Try to find content after YAML frontmatter
        yaml_pattern = re.compile(
            r'(---\s*\n.*?\n---\s*\n)(.*)',
            re.DOTALL
        )
        match = yaml_pattern.search(content)

        if match:
            return match.group(1), match.group(2).strip(), ""

        # No special structure found, use entire content
        return "", content.strip(), ""

    def format_file(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """
        Format a text file and save the result.
        For transcript files, only formats the transcription section.

        Args:
            input_path: Path to input file
            output_path: Path to output file (if None, overwrites input)

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            return False

        try:
            # Show initial status
            filename = os.path.basename(input_path)
            print(f"\n🔍 Found file: {filename}")

            # Read file
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                print("❌ File is empty")
                return False

            # Extract transcript content if it's a structured file
            prefix, transcript_text, suffix = self._extract_transcript_content(content)

            if not transcript_text:
                print("❌ No content found to format")
                return False

            # Format content with progress display
            formatted_content = self.format_text(transcript_text, show_progress=True)

            if not formatted_content:
                print("❌ Formatting failed")
                return False

            # Reconstruct the file with formatted content
            if prefix or suffix:
                # Structured file - preserve structure
                final_content = prefix + formatted_content + suffix
            else:
                # Plain text file - just use formatted content
                final_content = formatted_content

            # Determine output path
            if output_path is None:
                output_path = input_path

            # Write formatted content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_content)

            # Show completion screen
            self._show_completion(filename, output_path, len(transcript_text), len(formatted_content))
            return True

        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return False

    def _show_completion(self, filename: str, file_path: str, original_chars: int, formatted_chars: int):
        """Show completion screen in consistent UI style."""
        # Clear screen
        os.system('clear' if os.name != 'nt' else 'cls')

        print("___________________________________\n")
        print("✅ COMPLETE\n")
        print(f"FILE          {filename}")
        print(f"ORIGINAL      {original_chars:,} chars")
        print(f"FORMATTED     {formatted_chars:,} chars\n")
        print(f"Saved to {file_path}")
        print("___________________________________\n")


def get_formatter(ollama_model=None, ollama_api_url=None,
                  ollama_timeout=None, chunk_size=None, overlap_size=None):
    """
    Factory function to create configured FormattingService instance.
    Loads defaults from environment variables if not provided.

    Args:
        ollama_model: Ollama model name (default: from OLLAMA_MODEL env)
        ollama_api_url: Ollama API URL (default: from OLLAMA_API_URL env)
        ollama_timeout: Timeout in seconds (default: from OLLAMA_TIMEOUT env)
        chunk_size: Chunk size in characters (default: 1000)
        overlap_size: Overlap size in characters (default: 200)

    Returns:
        FormattingService: Configured service instance
    """
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=env_path)

    return FormattingService(
        ollama_model=ollama_model or os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        ollama_api_url=ollama_api_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate"),
        ollama_timeout=ollama_timeout or int(os.getenv("OLLAMA_TIMEOUT", "180")),
        chunk_size=chunk_size or 1000,
        overlap_size=overlap_size or 200
    )
