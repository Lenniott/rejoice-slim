# src/transcribe.py

# Suppress urllib3 SSL warnings before any imports
import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

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
import uuid
from typing import Optional, Dict, Any

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
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))  # Default 3 minutes for local LLMs
SAMPLE_RATE = 16000 # 16kHz is standard for Whisper

# Real-time chunking configuration
CHUNK_DURATION_SECONDS = float(os.getenv("CHUNK_DURATION_SECONDS", "10"))
SILENCE_DURATION_SECONDS = int(os.getenv("SILENCE_DURATION_SECONDS", "120"))

# Hardcoded advanced settings (optimized defaults)
CHUNK_OVERLAP_SECONDS = float(os.getenv("CHUNK_OVERLAP_SECONDS", "2.5"))
TRANSCRIPTION_WORKER_THREADS = int(os.getenv("TRANSCRIPTION_WORKER_THREADS", "2"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "4"))

# Calculate silence trigger chunks from duration
SILENCE_TRIGGER_CHUNKS = int(SILENCE_DURATION_SECONDS / CHUNK_DURATION_SECONDS)

# Audio device configuration
DEFAULT_MIC_DEVICE = int(os.getenv("DEFAULT_MIC_DEVICE", "-1"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress verbose logging from our libraries
logging.getLogger('audio_chunker').setLevel(logging.WARNING)
logging.getLogger('transcription_worker').setLevel(logging.WARNING)
logging.getLogger('vad_service').setLevel(logging.WARNING)

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
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_ollama(prompt_key, text_content, prompts):
    """Generic function to call Ollama with different prompts"""
    if prompt_key not in prompts:
        print(f"⚠️ Prompt '{prompt_key}' not found in prompts.json")
        return None
    
    prompt_template = prompts[prompt_key]["prompt"]
    # Send full transcript, not truncated
    prompt = prompt_template.format(text=text_content)
    
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for faster, more focused responses
                "top_p": 0.9,
                "max_tokens": 200  # Limit response length for speed
            }
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        
        # Get response and clean it up
        raw_response = json.loads(response.text)["response"].strip()
        
        # Remove thinking tags and extra content (more comprehensive)
        import re
        # Remove various thinking patterns
        cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\[thinking\].*?\[/thinking\]', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'^.*?(\{.*\}).*$', r'\1', cleaned, flags=re.DOTALL)  # Extract JSON if it exists
        cleaned = cleaned.strip()
        
        return cleaned if cleaned else None
        
    except requests.exceptions.Timeout:
        timeout_minutes = OLLAMA_TIMEOUT // 60
        timeout_seconds = OLLAMA_TIMEOUT % 60
        time_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
        print(f"⚠️ Ollama response timed out after {time_str}")
        print("💡 Tip: Use a faster model or increase timeout with: export OLLAMA_TIMEOUT=300")
        return None
    except requests.exceptions.ConnectionError:
        print(f"⚠️ Could not connect to Ollama for {prompt_key}")
        print("💡 Tip: Make sure Ollama is running with: ollama serve")
        return None
    except Exception as e:
        print(f"⚠️ Ollama error for {prompt_key}: {e}")
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

# Old individual LLM functions removed - now using combined_metadata

def list_audio_devices():
    """List all available audio input devices"""
    import sounddevice as sd
    devices = sd.query_devices()
    print("\nAvailable audio input devices:")
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']}")
            input_devices.append((i, device['name']))
    return input_devices

def get_combined_metadata_from_llm(text_content):
    """Gets filename, summary, and tags in one LLM call."""
    if not check_ollama_available():
        return None
    
    prompts = load_prompts()
    result = call_ollama("combined_metadata", text_content, prompts)
    
    if result:
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = result.strip()
            
            # Parse JSON response - handle extra data after JSON
            try:
                metadata = json.loads(json_str)
            except json.JSONDecodeError as e:
                if "Extra data" in str(e):
                    # Try to parse just the first valid JSON object
                    lines = json_str.split('\n')
                    json_lines = []
                    for line in lines:
                        json_lines.append(line)
                        try:
                            test_json = json.loads('\n'.join(json_lines))
                            metadata = test_json
                            break
                        except json.JSONDecodeError:
                            continue
                    else:
                        raise e
                else:
                    raise e
            
            # Validate required fields
            if all(key in metadata for key in ['filename', 'summary', 'tags']):
                # Clean up filename
                filename = metadata['filename'].strip()
                if not filename:
                    return None
                
                # Clean up tags
                tags = metadata['tags']
                if isinstance(tags, list):
                    tags = [tag.strip().lower() for tag in tags if tag.strip()]
                    tags = tags[:5]  # Limit to 5 tags max
                else:
                    tags = []
                
                return {
                    'filename': filename,
                    'summary': metadata['summary'].strip(),
                    'tags': tags
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing LLM metadata response: {e}")
            print(f"Raw response: {result[:200]}...")
            return None
    
    return None

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
    """Interactive settings menu with categories"""
    print("\n⚙️  Settings Menu")
    print("─" * 50)
    
    while True:
        print("\n📋 Settings Categories:")
        print("  1. 📝 Transcription (Whisper model, language)")
        print("  2. 📁 Output (Format, save path, auto-actions)")
        print("  3. 🤖 AI (Ollama model, auto-metadata)")
        print("  4. 🎤 Audio (Microphone device)")
        print("  5. ⚡ Performance (Chunking, auto-stop)")
        print("  6. 🚪 Exit")
        
        choice = input("\n👉 Choose a category (1-6): ").strip()
        
        if choice == "1":
            transcription_settings()
        elif choice == "2":
            output_settings()
        elif choice == "3":
            ai_settings()
        elif choice == "4":
            audio_settings()
        elif choice == "5":
            advanced_performance_settings()
        elif choice == "6":
            print("👋 Exiting settings...")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

def transcription_settings():
    """Transcription settings submenu"""
    while True:
        print(f"\n📝 Transcription Settings")
        print("─" * 30)
        print(f"Current Whisper Model: {WHISPER_MODEL}")
        print(f"Current Language: {WHISPER_LANGUAGE}")
        print(f"\n1. Change Whisper Model")
        print(f"2. Change Language")
        print(f"3. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-3): ").strip()
        
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
                print(f"\n📥 Downloading Whisper model '{new_model}'...")
                print("This may take a moment depending on the model size...")
                try:
                    import whisper
                    whisper.load_model(new_model)
                    update_env_setting("WHISPER_MODEL", new_model)
                    print(f"✅ Whisper model changed to: {new_model}")
                    print("✅ Model downloaded and ready to use")
                except Exception as e:
                    print(f"❌ Failed to download model: {e}")
                    print("⚠️ Model setting not updated")
        
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
            break
        else:
            print("❌ Invalid choice. Please select 1-3.")

def output_settings():
    """Output settings submenu"""
    while True:
        print(f"\n📁 Output Settings")
        print("─" * 20)
        print(f"Current Format: {OUTPUT_FORMAT}")
        print(f"Current Save Path: {SAVE_PATH}")
        print(f"Auto Copy: {'Yes' if AUTO_COPY else 'No'}")
        print(f"Auto Open: {'Yes' if AUTO_OPEN else 'No'}")
        print(f"Auto Metadata: {'Yes' if AUTO_METADATA else 'No'}")
        print(f"\n1. Change Output Format")
        print(f"2. Change Save Path")
        print(f"3. Toggle Auto Copy")
        print(f"4. Toggle Auto Open")
        print(f"5. Toggle Auto Metadata")
        print(f"6. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-6): ").strip()
        
        if choice == "1":
            format_choice = input("Choose output format (md/txt): ").strip().lower()
            if format_choice in ["md", "txt"]:
                update_env_setting("OUTPUT_FORMAT", format_choice)
                print(f"✅ Output format changed to: {format_choice}")
                print("⚠️ Restart the script to use the new format")
        
        elif choice == "2":
            new_path = input(f"Enter new save path [{SAVE_PATH}]: ").strip()
            if new_path:
                os.makedirs(new_path, exist_ok=True)
                update_env_setting("SAVE_PATH", new_path)
                print(f"✅ Save path changed to: {new_path}")
                print("⚠️ Restart the script to use the new path")
        
        elif choice == "3":
            new_setting = input("Auto copy to clipboard? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_COPY", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto copy changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "4":
            new_setting = input("Auto open file? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_OPEN", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto open changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "5":
            new_setting = input("Auto generate AI metadata? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_METADATA", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto metadata changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "6":
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

def ai_settings():
    """AI settings submenu"""
    while True:
        timeout_minutes = OLLAMA_TIMEOUT // 60
        timeout_seconds = OLLAMA_TIMEOUT % 60
        timeout_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
        
        print(f"\n🤖 AI Settings")
        print("─" * 15)
        print(f"Current Ollama Model: {OLLAMA_MODEL}")
        print(f"Auto Metadata: {'Yes' if AUTO_METADATA else 'No'}")
        print(f"Ollama Timeout: {timeout_str}")
        print(f"\n1. Change Ollama Model")
        print(f"2. Toggle Auto Metadata")
        print(f"3. Change Ollama Timeout")
        print(f"4. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-4): ").strip()
        
        if choice == "1":
            print("\nSuggested Ollama Models:")
            print("  • gemma3:4b (recommended)")
            print("  • llama3 (good alternative)")
            print("  • qwen3:0.6b (fast)")
            print("  • phi3")
            print("  • gemma")
            
            new_model = input("\nEnter Ollama model name: ").strip()
            if new_model:
                update_env_setting("OLLAMA_MODEL", new_model)
                print(f"✅ Ollama model changed to: {new_model}")
                print("⚠️ Restart the script to use the new model")
        
        elif choice == "2":
            new_setting = input("Auto generate AI metadata? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_METADATA", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto metadata changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "3":
            current_timeout = int(os.getenv('OLLAMA_TIMEOUT', '180'))
            print(f"\nCurrent timeout: {current_timeout} seconds")
            print("Recommended timeouts:")
            print("  • 60s  - Fast models (gemma3:270m, qwen3:0.6b)")
            print("  • 180s - Medium models (gemma3:4b, llama3)")  
            print("  • 300s - Large models (llama3:70b)")
            
            new_timeout = input(f"Enter timeout in seconds (30-600) [current: {current_timeout}]: ").strip()
            try:
                timeout = int(new_timeout) if new_timeout else current_timeout
                if 30 <= timeout <= 600:
                    update_env_setting("OLLAMA_TIMEOUT", str(timeout))
                    timeout_minutes = timeout // 60
                    timeout_seconds = timeout % 60
                    timeout_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
                    print(f"✅ Ollama timeout changed to: {timeout_str}")
                    print("⚠️ Restart the script to use the new setting")
                else:
                    print("❌ Timeout must be between 30 and 600 seconds (10 minutes)")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "4":
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

def audio_settings():
    """Audio settings submenu"""
    while True:
        print(f"\n🎤 Audio Settings")
        print("─" * 18)
        print(f"Current Microphone Device: {DEFAULT_MIC_DEVICE if DEFAULT_MIC_DEVICE != -1 else 'System Default'}")
        print(f"\n1. Change Microphone Device")
        print(f"2. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-2): ").strip()
        
        if choice == "1":
            print("\n--- Microphone Device Selection ---")
            devices = list_audio_devices()
            if devices:
                print(f"\nCurrent device: {DEFAULT_MIC_DEVICE if DEFAULT_MIC_DEVICE != -1 else 'System Default'}")
                device_choice = input("Enter device number (-1 for system default): ").strip()
                try:
                    device_num = int(device_choice)
                    if device_num == -1 or any(device_num == dev[0] for dev in devices):
                        update_env_setting("DEFAULT_MIC_DEVICE", str(device_num))
                        print(f"✅ Microphone device changed to: {device_num if device_num != -1 else 'System Default'}")
                        print("⚠️ Restart the script to use the new device")
                    else:
                        print("❌ Invalid device number")
                except ValueError:
                    print("❌ Please enter a valid number")
            else:
                print("❌ No audio input devices found")
        
        elif choice == "2":
            break
        else:
            print("❌ Invalid choice. Please select 1-2.")

def advanced_performance_settings():
    """Advanced performance settings submenu"""
    while True:
        print(f"\n⚡ Performance Settings")
        print("─" * 25)
        print(f"Chunk Duration: {os.getenv('CHUNK_DURATION_SECONDS', '10')} seconds")
        print(f"No Speech Detection: {os.getenv('SILENCE_DURATION_SECONDS', '120')} seconds")
        print(f"\n1. Change Chunk Duration")
        print(f"2. Change No Speech Detection Duration")
        print(f"3. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-3): ").strip()
        
        if choice == "1":
            current_duration = int(os.getenv('CHUNK_DURATION_SECONDS', '10'))
            new_duration = input(f"Enter chunk duration in seconds (5-30) [current: {current_duration}]: ").strip()
            try:
                duration = int(new_duration) if new_duration else current_duration
                if 5 <= duration <= 30:
                    if duration < 8:
                        print("ℹ️ Shorter duration = more frequent updates")
                    elif duration > 15:
                        print("ℹ️ Longer duration = less frequent updates")
                    update_env_setting("CHUNK_DURATION_SECONDS", str(duration))
                    print(f"✅ Chunk duration changed to: {duration} seconds")
                    print("⚠️ Restart the script to use the new setting")
                else:
                    print("❌ Duration must be between 5 and 30 seconds")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "2":
            current_silence = int(os.getenv('SILENCE_DURATION_SECONDS', '120'))
            new_silence = input(f"Enter no speech detection duration in seconds (30-300) [current: {current_silence}]: ").strip()
            try:
                silence = int(new_silence) if new_silence else current_silence
                if 30 <= silence <= 300:
                    minutes = silence // 60
                    seconds = silence % 60
                    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                    update_env_setting("SILENCE_DURATION_SECONDS", str(silence))
                    print(f"✅ No speech detection changed to: {silence} seconds ({time_str})")
                    print("⚠️ Restart the script to use the new setting")
                else:
                    print("❌ Duration must be between 30 and 300 seconds")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "3":
            break
        else:
            print("❌ Invalid choice. Please select 1-3.")

def record_audio_chunked(device_override: Optional[int] = None) -> Optional[str]:
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
                chunks_processed = 0
                for chunk in chunker.get_ready_chunks():
                    # Send to transcription workers
                    worker_pool.add_chunk(chunk)
                    chunks_processed += 1
                    
                    # Send to VAD service
                    vad_service.analyze_chunk(chunk)
                
                # Chunk processing happens silently in background
                
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
        
        # Start audio recording with configured device (or override)
        device = device_override if device_override is not None else (None if DEFAULT_MIC_DEVICE == -1 else DEFAULT_MIC_DEVICE)
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, device=device)
        stream.start()
        
        # Wait for user to press Enter
        input()
        
        # Stop recording
        stream.stop()
        stream.close()
        
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
        
        # If no chunks were processed, fall back to transcribing the full audio
        if not transcript.strip() and audio_data:
            full_audio = np.concatenate(audio_data, axis=0)
            if WHISPER_LANGUAGE and WHISPER_LANGUAGE.lower() != "auto":
                result = whisper_model.transcribe(full_audio, fp16=False, language=WHISPER_LANGUAGE)
            else:
                result = whisper_model.transcribe(full_audio, fp16=False)
            transcript = result["text"].strip()
        
        if not transcript.strip():
            print("No speech detected in the audio.")
            return None
        
        # Transcription complete - no need to show chunk statistics
        
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
    """Handle file opening based on settings"""
    
    # Determine actions based on args or auto settings
    should_open = args.open if hasattr(args, 'open') and args.open is not None else AUTO_OPEN
    
    # Open file - only if explicitly enabled
    if should_open:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([opener, full_path])
        print("📂 File opened.")
    elif not hasattr(args, 'open') or args.open is None:
        # Only ask if AUTO_OPEN is not explicitly set to false
        if not AUTO_OPEN:
            # Don't ask, just skip
            pass
        else:
            # Ask user
            if input("📂 Open the file? (y/n): ").lower() == 'y':
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.run([opener, full_path])

def main(args=None):
    # Set defaults if no args provided
    if args is None:
        args = type('Args', (), {})()
    
    # 1. Record Audio with real-time chunking and transcription
    device_override = args.device if hasattr(args, 'device') and args.device is not None else None
    transcribed_text = record_audio_chunked(device_override)
    if not transcribed_text:
        return

    # 2. Deduplicate transcript to remove any repetition from chunk overlap
    transcribed_text = deduplicate_transcript(transcribed_text)

    print("\n--- TRANSCRIPT ---")
    print(transcribed_text)
    print("--------------------")

    # 3. Copy to clipboard immediately (before LLM processing)
    if AUTO_COPY:
        pyperclip.copy(transcribed_text)
        print("📋 Transcription copied to clipboard.")

    # 4. Save file immediately with timestamp name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    unique_id = str(uuid.uuid4())[:6]
    temp_filename = f"{timestamp}_{unique_id}.{OUTPUT_FORMAT}"
    temp_path = os.path.join(SAVE_PATH, temp_filename)
    
    # Create basic content
    templates = load_templates()
    now = datetime.now()
    created_date = now.strftime("%Y-%m-%d")
    created_time = now.strftime("%H:%M")
    
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
    
    # Save immediately
    with open(temp_path, "w") as f:
        f.write(content)
    print(f"✅ Successfully saved transcript to: {temp_path}")

    # 4. Generate metadata and rename file (if Ollama available)
    ollama_available = check_ollama_available()
    final_path = temp_path  # Default to temp path
    
    if ollama_available:
        metadata = get_combined_metadata_from_llm(transcribed_text)
        
        if metadata:
            # Create final filename with unique ID preserved
            final_filename = f"{metadata['filename']}_{timestamp}_{unique_id}.{OUTPUT_FORMAT}"
            final_path = os.path.join(SAVE_PATH, final_filename)
            
            # Rename file
            os.rename(temp_path, final_path)
            
            # Update file with metadata
            try:
                with open(final_path, "r") as f:
                    current_content = f.read()
                
                # Replace summary
                updated_content = current_content.replace(
                    "[Summary will be generated if requested]", 
                    metadata['summary']
                )
                
                # Add tags
                if metadata['tags']:
                    tags_to_add = "\n".join([f"  - {tag}" for tag in metadata['tags']])
                    updated_content = updated_content.replace(
                        "tags:\n  - voice-note",
                        f"tags:\n  - voice-note\n{tags_to_add}"
                    )
                
                with open(final_path, "w") as f:
                    f.write(updated_content)
                
                print(f"📝 Summary: {metadata['summary']}")
                if metadata['tags']:
                    print(f"🏷️  Tags: {', '.join(metadata['tags'])}")
                print(f"✅ File renamed to: {final_path}")
                
            except Exception as e:
                print(f"⚠️ Error updating file with metadata: {e}")
        else:
            print("⚠️ Could not generate metadata, keeping original filename")
    else:
        print("ℹ️  Ollama not available - keeping timestamp filename")

    # 5. Handle post-transcription actions
    handle_post_transcription_actions(transcribed_text, final_path, ollama_available, args)

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
    parser.add_argument('--device', type=int, 
                       help='Override default mic device for this recording')
    
    # Set defaults to None so we can detect when they're not specified
    parser.set_defaults(copy=None, open=None, metadata=None)
    
    args = parser.parse_args()
    
    if not all([SAVE_PATH, OUTPUT_FORMAT, WHISPER_MODEL, OLLAMA_MODEL]):
        print("❌ Configuration is missing. Please run the setup.sh script first.")
    elif args.settings:
        settings_menu()
    else:
        main(args)