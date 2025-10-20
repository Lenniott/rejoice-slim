# src/transcribe.py

import os
import sys
import subprocess
import sounddevice as sd
import numpy as np
import whisper
import requests
import json
from scipy.io.wavfile import write
from datetime import datetime
from dotenv import load_dotenv
import pyperclip
import tempfile
import argparse
import threading
import queue
import logging
from typing import Optional

# Import our new chunking components
from audio_chunker import AudioChunker
from transcription_worker import TranscriptionWorkerPool
from vad_service import VADService

# --- CONFIGURATION ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
SAVE_PATH = os.getenv("SAVE_PATH")
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "md")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")
AUTO_COPY = os.getenv("AUTO_COPY", "false").lower() == "true"
AUTO_OPEN = os.getenv("AUTO_OPEN", "false").lower() == "true" 
AUTO_METADATA = os.getenv("AUTO_METADATA", "false").lower() == "true"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
SAMPLE_RATE = 16000 # 16kHz is standard for Whisper

# Real-time chunking configuration
CHUNK_DURATION_SECONDS = float(os.getenv("CHUNK_DURATION_SECONDS", "10"))
CHUNK_OVERLAP_SECONDS = float(os.getenv("CHUNK_OVERLAP_SECONDS", "1"))
TRANSCRIPTION_WORKER_THREADS = int(os.getenv("TRANSCRIPTION_WORKER_THREADS", "2"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
SILENCE_TRIGGER_CHUNKS = int(os.getenv("SILENCE_TRIGGER_CHUNKS", "30"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_prompts():
    """Load prompts from prompts.json file"""
    prompts_path = os.path.join(os.path.dirname(__file__), '..', 'prompts.json')
    try:
        with open(prompts_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load prompts file: {e}")
        return {}

def load_templates():
    """Load templates from templates.json file"""
    templates_path = os.path.join(os.path.dirname(__file__), '..', 'templates.json')
    try:
        with open(templates_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load templates file: {e}")
        return {}

def check_ollama_available():
    """Check if Ollama is available and running"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        return response.status_code == 200
    except:
        return False

def call_ollama(prompt_key, text_content, prompts):
    """Generic function to call Ollama with different prompts"""
    if prompt_key not in prompts:
        print(f"⚠️ Prompt '{prompt_key}' not found in prompts.json")
        return None
    
    prompt_template = prompts[prompt_key]["prompt"]
    prompt = prompt_template.format(text=text_content[:1000])
    
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        # Get response and clean it up
        raw_response = json.loads(response.text)["response"].strip()
        
        # Remove thinking tags and extra content
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        cleaned = cleaned.strip()
        
        return cleaned if cleaned else None
        
    except Exception as e:
        print(f"⚠️ Could not connect to Ollama for {prompt_key} ({e})")
        return None

def deduplicate_transcript(transcript: str) -> str:
    """
    Remove repeated phrases that might occur due to chunk overlap.
    
    Args:
        transcript: The complete transcript text
        
    Returns:
        str: Deduplicated transcript
    """
    words = transcript.split()
    if len(words) < 10:  # Don't process very short transcripts
        return transcript
    
    # Look for repeated phrases of 2-5 words
    for phrase_length in range(5, 1, -1):
        i = 0
        while i < len(words) - phrase_length * 2:
            phrase1 = words[i:i + phrase_length]
            phrase2 = words[i + phrase_length:i + phrase_length * 2]
            
            if phrase1 == phrase2:
                # Remove the duplicate phrase
                words = words[:i + phrase_length] + words[i + phrase_length * 2:]
                # Don't advance i, check the same position again
                continue
            i += 1
    
    return ' '.join(words)

def get_filename_from_llm(text_content):
    """Asks Ollama for a concise filename, falls back to 'voice_note' if unavailable."""
    if not check_ollama_available():
        return "voice_note"
    
    prompts = load_prompts()
    result = call_ollama("filename_generation", text_content, prompts)
    
    if result:
        # Extract only alphanumeric words and underscores, limit length
        import re
        words = re.findall(r'[a-zA-Z0-9_]+', result)
        if words:
            # Take first 5 words max, join with underscores
            filename = '_'.join(words[:5])
            # Limit total length to 50 characters to avoid filesystem issues
            filename = filename[:50].lower()
            return filename if filename else "voice_note"
    
    return "voice_note"

def get_summary_from_llm(text_content):
    """Asks Ollama for a one-sentence summary."""
    if not check_ollama_available():
        return "A voice note transcription."
    
    prompts = load_prompts()
    result = call_ollama("summary_generation", text_content, prompts)
    return result if result else "A voice note transcription."

def get_tags_from_llm(text_content):
    """Asks Ollama for relevant tags."""
    if not check_ollama_available():
        return []
    
    prompts = load_prompts()
    result = call_ollama("tag_generation", text_content, prompts)
    
    if result:
        # Clean up tags: split by comma, clean whitespace, remove empty
        tags = [tag.strip().lower() for tag in result.split(',') if tag.strip()]
        return tags[:5]  # Limit to 5 tags max
    
    return []

def update_env_setting(key, value):
    """Update a setting in the .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    
    # Read current .env content
    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
    
    # Update or add the setting
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}='{value}'\n"
            updated = True
            break
    
    if not updated:
        lines.append(f"{key}='{value}'\n")
    
    # Write back to file
    with open(env_path, 'w') as f:
        f.writelines(lines)

def settings_menu():
    """Interactive settings menu"""
    print("⚙️ Settings Menu")
    print("================")
    
    while True:
        print(f"\nCurrent Settings:")
        print(f"1. Whisper Model: {WHISPER_MODEL}")
        print(f"2. Whisper Language: {WHISPER_LANGUAGE}")
        print(f"3. Ollama Model: {OLLAMA_MODEL}")
        print(f"4. Output Format: {OUTPUT_FORMAT}")
        print(f"5. Save Path: {SAVE_PATH}")
        print(f"6. Auto Copy: {'Yes' if AUTO_COPY else 'No'}")
        print(f"7. Auto Open: {'Yes' if AUTO_OPEN else 'No'}")
        print(f"8. Auto Metadata: {'Yes' if AUTO_METADATA else 'No'}")
        print(f"9. Exit Settings")
        
        choice = input("\nWhat would you like to change? (1-9): ").strip()
        
        if choice == "1":
            print("\nAvailable Whisper Models:")
            models = ["tiny", "base", "small", "medium", "large"]
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            
            model_choice = input(f"\nChoose model (1-{len(models)}) or enter custom name: ").strip()
            
            if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
                new_model = models[int(model_choice) - 1]
            else:
                new_model = model_choice
            
            if new_model:
                update_env_setting("WHISPER_MODEL", new_model)
                print(f"✅ Whisper model changed to: {new_model}")
                print("⚠️ Restart the script to use the new model")
        
        elif choice == "2":
            print("\nCommon languages:")
            print("  • en (English)")
            print("  • es (Spanish)")
            print("  • fr (French)")
            print("  • de (German)")
            print("  • it (Italian)")
            print("  • pt (Portuguese)")
            print("  • auto (automatic detection)")
            
            new_language = input("\nEnter language code: ").strip().lower()
            if new_language:
                update_env_setting("WHISPER_LANGUAGE", new_language)
                print(f"✅ Whisper language changed to: {new_language}")
                print("⚠️ Restart the script to use the new language")
        
        elif choice == "3":
            print("\nSuggested Ollama Models:")
            print("  • llama3 (recommended)")
            print("  • qwen3:0.6b (fast)")
            print("  • phi3")
            print("  • gemma")
            
            new_model = input("\nEnter Ollama model name: ").strip()
            if new_model:
                update_env_setting("OLLAMA_MODEL", new_model)
                print(f"✅ Ollama model changed to: {new_model}")
                print("⚠️ Restart the script to use the new model")
        
        elif choice == "4":
            format_choice = input("Choose output format (md/txt): ").strip().lower()
            if format_choice in ["md", "txt"]:
                update_env_setting("OUTPUT_FORMAT", format_choice)
                print(f"✅ Output format changed to: {format_choice}")
                print("⚠️ Restart the script to use the new format")
        
        elif choice == "5":
            new_path = input(f"Enter new save path [{SAVE_PATH}]: ").strip()
            if new_path:
                # Create directory if it doesn't exist
                os.makedirs(new_path, exist_ok=True)
                update_env_setting("SAVE_PATH", new_path)
                print(f"✅ Save path changed to: {new_path}")
                print("⚠️ Restart the script to use the new path")
        
        elif choice == "6":
            new_setting = input("Auto copy to clipboard? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_COPY", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto copy changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "7":
            new_setting = input("Auto open file? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_OPEN", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto open changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "8":
            new_setting = input("Auto generate AI metadata? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_METADATA", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto metadata changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "9":
            print("👋 Exiting settings...")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-9.")

def record_audio_chunked() -> Optional[str]:
    """
    Records audio with real-time chunking and transcription.
    
    Returns:
        str or None: Complete assembled transcript, or None if recording failed
    """
    print("🔴 Recording... Press Enter to stop.")
    
    # Initialize components
    chunker = AudioChunker(
        chunk_duration_seconds=CHUNK_DURATION_SECONDS,
        overlap_seconds=CHUNK_OVERLAP_SECONDS,
        sample_rate=SAMPLE_RATE
    )
    
    # Load Whisper model
    print("🤫 Loading Whisper model...")
    whisper_model = whisper.load_model(WHISPER_MODEL)
    
    # Initialize worker pool
    worker_pool = TranscriptionWorkerPool(
        whisper_model=whisper_model,
        whisper_language=WHISPER_LANGUAGE,
        num_workers=TRANSCRIPTION_WORKER_THREADS,
        max_retry_attempts=MAX_RETRY_ATTEMPTS
    )
    
    # Initialize VAD service
    vad_service = VADService(
        silence_threshold_chunks=SILENCE_TRIGGER_CHUNKS,
        chunk_duration_seconds=CHUNK_DURATION_SECONDS
    )
    
    # Threading control
    recording_event = threading.Event()
    recording_event.set()
    stop_event = threading.Event()
    
    # Audio data collection
    audio_data = []
    
    def audio_callback(indata, frames, time, status):
        """Callback for sounddevice audio stream."""
        if status:
            print(status, file=sys.stderr)
        
        if recording_event.is_set():
            # Convert 2D array (channels x samples) to 1D array (samples)
            audio_1d = indata.flatten()
            audio_data.append(audio_1d.copy())
            chunker.add_audio_data(audio_1d.copy())
    
    def process_ready_chunks():
        """Process chunks as they become available."""
        while recording_event.is_set():
            try:
                for chunk in chunker.get_ready_chunks():
                    # Send to transcription workers
                    worker_pool.add_chunk(chunk)
                    
                    # Send to VAD service
                    vad_service.analyze_chunk(chunk)
                
                # Small delay to prevent busy waiting
                threading.Event().wait(0.1)
            except Exception as e:
                logging.error(f"Error processing chunks: {e}")
                break
    
    try:
        # Start worker pool
        worker_pool.start()
        vad_service.start_recording()
        
        # Start chunk processing thread
        chunk_thread = threading.Thread(target=process_ready_chunks, daemon=True)
        chunk_thread.start()
        
        # Start audio recording
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            # Wait for user to press Enter
            input()
        
        print("\nRecording finished. Processing final chunk...")
        
        # Stop recording
        recording_event.clear()
        
        # Process final partial chunk
        final_chunk = chunker.get_final_chunk()
        if final_chunk is not None:
            worker_pool.add_chunk(final_chunk)
            vad_service.analyze_chunk(final_chunk)
        
        # Wait for all workers to finish
        print("⏳ Finalizing transcription...")
        worker_pool.stop()
        vad_service.stop_recording()
        
        # Get assembled transcript
        transcript = worker_pool.get_assembled_transcript()
        
        if not transcript.strip():
            print("No speech detected in the audio.")
            return None
        
        # Print statistics
        stats = worker_pool.get_stats()
        print(f"✅ Transcription complete: {stats['total_chunks_processed']} chunks processed, "
              f"{stats['total_chunks_failed']} failed")
        
        return transcript
        
    except Exception as e:
        print(f"An error occurred during recording: {e}")
        logging.error(f"Recording error: {e}")
        return None
    finally:
        # Ensure cleanup
        recording_event.clear()
        if 'worker_pool' in locals():
            worker_pool.stop()
        if 'vad_service' in locals():
            vad_service.stop_recording()

def handle_post_transcription_actions(transcribed_text, full_path, ollama_available, args):
    """Handle clipboard, file opening, and AI metadata generation based on settings"""
    
    # Determine actions based on args or auto settings
    should_copy = args.copy if hasattr(args, 'copy') and args.copy is not None else AUTO_COPY
    should_open = args.open if hasattr(args, 'open') and args.open is not None else AUTO_OPEN
    should_metadata = args.metadata if hasattr(args, 'metadata') and args.metadata is not None else AUTO_METADATA
    
    # Copy to clipboard
    if should_copy:
        pyperclip.copy(transcribed_text)
        print("📋 Transcription copied to clipboard.")
    elif not hasattr(args, 'copy') or args.copy is None:
        # Only ask if not specified via command line AND auto-copy is disabled
        if not AUTO_COPY and input("\n📋 Copy transcription to clipboard? (y/n): ").lower() == 'y':
            pyperclip.copy(transcribed_text)
            print("Transcription copied to clipboard.")
    
    # Open file
    if should_open:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([opener, full_path])
        print("📂 File opened.")
    elif not hasattr(args, 'open') or args.open is None:
        # Only ask if not specified via command line AND auto-open is disabled
        if not AUTO_OPEN and input("📂 Open the file? (y/n): ").lower() == 'y':
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.run([opener, full_path])
    
    # Generate AI metadata
    if ollama_available and should_metadata:
        print("🧠 Generating AI summary and tags...")
        summary = get_summary_from_llm(transcribed_text)
        tags = get_tags_from_llm(transcribed_text)
        
        # Update the file with summary and tags
        try:
            with open(full_path, "r") as f:
                current_content = f.read()
            
            # Replace the placeholder summary
            updated_content = current_content.replace(
                "[Summary will be generated if requested]", 
                summary
            )
            
            # Add tags if we have them
            if tags:
                tags_to_add = "\n".join([f"  - {tag}" for tag in tags])
                updated_content = updated_content.replace(
                    "tags:\n  - voice-note",
                    f"tags:\n  - voice-note\n{tags_to_add}"
                )
            
            with open(full_path, "w") as f:
                f.write(updated_content)
            
            print(f"📝 Summary: {summary}")
            if tags:
                print(f"🏷️  Tags: {', '.join(tags)}")
            print("✅ File updated with AI-generated metadata!")
            
        except Exception as e:
            print(f"⚠️ Error updating file: {e}")
            
    elif ollama_available and not should_metadata and (not hasattr(args, 'metadata') or args.metadata is None):
        # Only ask if not specified via command line AND auto-metadata is disabled
        if not AUTO_METADATA and input("🤖 Generate AI summary and tags? (y/n): ").lower() == 'y':
            print("🧠 Generating summary and tags...")
            summary = get_summary_from_llm(transcribed_text)
            tags = get_tags_from_llm(transcribed_text)
            
            try:
                with open(full_path, "r") as f:
                    current_content = f.read()
                
                updated_content = current_content.replace(
                    "[Summary will be generated if requested]", 
                    summary
                )
                
                if tags:
                    tags_to_add = "\n".join([f"  - {tag}" for tag in tags])
                    updated_content = updated_content.replace(
                        "tags:\n  - voice-note",
                        f"tags:\n  - voice-note\n{tags_to_add}"
                    )
                
                with open(full_path, "w") as f:
                    f.write(updated_content)
                
                print(f"\n📝 Summary: {summary}")
                if tags:
                    print(f"🏷️  Tags: {', '.join(tags)}")
                print("✅ File updated with AI-generated metadata!")
                
            except Exception as e:
                print(f"⚠️ Error updating file: {e}")
                
    elif not ollama_available:
        print("ℹ️  Ollama not available - AI features disabled. Transcription saved successfully!")

def main(args=None):
    # Set defaults if no args provided
    if args is None:
        args = type('Args', (), {})()
    
    # 1. Record Audio with real-time chunking and transcription
    transcribed_text = record_audio_chunked()
    if not transcribed_text:
        return

    # 2. Deduplicate transcript to remove any repetition from chunk overlap
    transcribed_text = deduplicate_transcript(transcribed_text)

    print("\n--- TRANSCRIPT ---")
    print(transcribed_text)
    print("--------------------")

    # 2. Quick filename generation and save immediately
    ollama_available = check_ollama_available()
    if ollama_available:
        print("🧠 Generating filename...")
    else:
        print("📝 Creating filename (Ollama not available)...")
    
    base_name = get_filename_from_llm(transcribed_text)
    timestamp = datetime.now().strftime("%d%m%y_%H%M")
    final_filename = f"{base_name}_{timestamp}.{OUTPUT_FORMAT}"
    full_path = os.path.join(SAVE_PATH, final_filename)

    # 3. Create initial file with basic template (no summary/tags yet)
    templates = load_templates()
    now = datetime.now()
    created_date = now.strftime("%Y-%m-%d")
    created_time = now.strftime("%H:%M")
    
    # Create basic content without summary/tags
    template_key = f"{OUTPUT_FORMAT}_template"
    
    if template_key in templates:
        content = templates[template_key].format(
            created_date=created_date,
            created_time=created_time,
            summary="[Summary will be generated if requested]",
            transcription=transcribed_text,
            tags_section="\n  - voice-note",
            tags_inline="voice-note"
        )
    else:
        # Fallback to simple format
        content = f"# Transcription: {now.strftime('%d %B %Y, %H:%M')}\n\n{transcribed_text}"
    
    # 4. Save the transcript immediately
    with open(full_path, "w") as f:
        f.write(content)
    print(f"✅ Successfully saved transcript to: {full_path}")

    # 5. Handle post-transcription actions
    handle_post_transcription_actions(transcribed_text, full_path, ollama_available, args)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Voice transcription tool')
    parser.add_argument('-s', '--settings', action='store_true', 
                       help='Open settings menu to change configuration')
    parser.add_argument('--copy', action='store_true', dest='copy',
                       help='Auto copy transcription to clipboard')
    parser.add_argument('--no-copy', action='store_false', dest='copy',
                       help='Do not copy transcription to clipboard')
    parser.add_argument('--open', action='store_true', dest='open',
                       help='Auto open the transcription file')
    parser.add_argument('--no-open', action='store_false', dest='open',
                       help='Do not open the transcription file')
    parser.add_argument('--metadata', action='store_true', dest='metadata',
                       help='Auto generate AI summary and tags')
    parser.add_argument('--no-metadata', action='store_false', dest='metadata',
                       help='Do not generate AI summary and tags')
    
    # Set defaults to None so we can detect when they're not specified
    parser.set_defaults(copy=None, open=None, metadata=None)
    
    args = parser.parse_args()
    
    if not all([SAVE_PATH, OUTPUT_FORMAT, WHISPER_MODEL, OLLAMA_MODEL]):
        print("❌ Configuration is missing. Please run the setup.sh script first.")
    elif args.settings:
        settings_menu()
    else:
        main(args)