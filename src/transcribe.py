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
import signal
import select
import termios
import tty
from typing import Optional, Dict, Any

# Import our new chunking components
from audio_chunker import AudioChunker
from transcription_worker import TranscriptionWorkerPool
from vad_service import VADService

# Import our new ID-based transcript management
from transcript_manager import TranscriptFileManager
from id_generator import TranscriptIDGenerator
from file_header import TranscriptHeader

# Import summarization service
from summarization_service import SummarizationService

# --- CONFIGURATION ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
SAVE_PATH = os.getenv("SAVE_PATH")
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "md")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")
OLLAMA_MAX_CONTENT_LENGTH = int(os.getenv("OLLAMA_MAX_CONTENT_LENGTH", "32000"))  # Character limit for AI processing
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
                    tags = [tag.strip().lower().replace(' ', '-') for tag in tags if tag.strip()]
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
    
    # Also update the current process environment
    os.environ[key] = value

def settings_menu():
    """Interactive settings menu with categories"""
    try:
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
    except KeyboardInterrupt:
        if sys.platform == "darwin":  # macOS
            print("\n\n👋 Settings menu cancelled by user (Ctrl+C).")
        else:
            print("\n\n👋 Settings menu cancelled by user.")
    except EOFError:
        print("\n\n👋 Settings menu closed.")
    except Exception as e:
        print(f"\n❌ Error in settings menu: {e}")

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
        # Read current values dynamically from environment
        current_model = os.getenv('OLLAMA_MODEL', 'gemma3:270m')
        current_metadata = os.getenv('AUTO_METADATA', 'false').lower() == 'true'
        current_timeout = int(os.getenv('OLLAMA_TIMEOUT', '180'))
        current_max_length = int(os.getenv('OLLAMA_MAX_CONTENT_LENGTH', '32000'))
        
        timeout_minutes = current_timeout // 60
        timeout_seconds = current_timeout % 60
        timeout_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
        
        print(f"\n🤖 AI Settings")
        print("─" * 15)
        print(f"Current Ollama Model: {current_model}")
        print(f"Auto Metadata: {'Yes' if current_metadata else 'No'}")
        print(f"Ollama Timeout: {timeout_str}")
        print(f"Max Content Length: {current_max_length:,} characters")
        print(f"\n1. Change Ollama Model")
        print(f"2. Toggle Auto Metadata")
        print(f"3. Change Ollama Timeout")
        print(f"4. Change Max Content Length")
        print(f"5. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-5): ").strip()
        
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
            print(f"\nCurrent max content length: {current_max_length:,} characters")
            print("Recommended character limits:")
            print("  • 8,000   - Conservative (original default)")
            print("  • 32,000  - Balanced (new default)")
            print("  • 64,000  - For powerful setups")
            print("  • 128,000 - Maximum (requires robust hardware)")
            
            new_length = input(f"Enter max content length (1000-200000) [current: {current_max_length:,}]: ").strip()
            try:
                length = int(new_length.replace(',', '')) if new_length else current_max_length
                if 1000 <= length <= 200000:
                    update_env_setting("OLLAMA_MAX_CONTENT_LENGTH", str(length))
                    print(f"✅ Max content length changed to: {length:,} characters")
                    print("⚠️ Restart the script to use the new setting")
                else:
                    print("❌ Length must be between 1,000 and 200,000 characters")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "5":
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

def setup_keyboard_handler():
    """Set up terminal for non-blocking keyboard input."""
    if sys.platform != "win32":  # Unix-like systems
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setraw(sys.stdin.fileno())
            return fd, old_settings
        except (OSError, termios.error):
            # Terminal doesn't support raw mode (e.g., in some IDEs)
            return None, None
    return None, None

def restore_keyboard_handler(fd, old_settings):
    """Restore terminal settings."""
    if fd is not None and old_settings is not None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def check_keyboard_input(fd):
    """Check for keyboard input without blocking."""
    if sys.platform == "win32":
        # Windows implementation (simplified)
        try:
            import msvcrt
            return msvcrt.kbhit()
        except ImportError:
            # Fallback if msvcrt not available
            return False
    else:
        # Unix-like systems
        try:
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        except (OSError, ValueError):
            # Fallback if select doesn't work (e.g., in some IDEs)
            return False

def get_keyboard_char():
    """Get a keyboard character."""
    if sys.platform == "win32":
        try:
            import msvcrt
            return msvcrt.getch().decode('utf-8')
        except ImportError:
            return sys.stdin.readline().strip()
    else:
        return sys.stdin.read(1)

# Global cancellation state - accessible from signal handler
_global_recording_state = {
    'recording_event': None,
    'cancelled': None,
    'original_handler': None
}

def _global_signal_handler(signum, frame):
    """Global signal handler for Ctrl+C."""
    try:
        if sys.platform == "darwin":  # macOS
            print("\n🚫 Recording cancelled by user (Ctrl+C).")
        else:
            print("\n🚫 Recording cancelled by user.")
        
        # Set global cancellation state
        if _global_recording_state['cancelled']:
            _global_recording_state['cancelled'].set()
        if _global_recording_state['recording_event']:
            _global_recording_state['recording_event'].clear()
            
        # Force flush to ensure message is shown immediately
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        # Ensure we always set the cancelled flag even if printing fails
        if _global_recording_state['cancelled']:
            _global_recording_state['cancelled'].set()
        if _global_recording_state['recording_event']:
            _global_recording_state['recording_event'].clear()

def record_audio_chunked(device_override: Optional[int] = None) -> Optional[str]:
    """
    Records audio with real-time chunking and transcription.
    
    Returns:
        str or None: Complete assembled transcript, or None if recording failed or cancelled
    """
    # Platform-specific messaging for keyboard shortcuts
    if sys.platform == "darwin":  # macOS
        print("🔴 Recording... Press Enter to stop, Ctrl+C (^C) to cancel.")
    else:
        print("🔴 Recording... Press Enter to stop, Ctrl+C to cancel.")
    
    # Initialize components
    chunker = AudioChunker(
        chunk_duration_seconds=CHUNK_DURATION_SECONDS,
        overlap_seconds=CHUNK_OVERLAP_SECONDS,
        sample_rate=SAMPLE_RATE
    )
    
    # Load Whisper model with clean interruption handling
    print("🤫 Loading Whisper model...")
    try:
        whisper_model = whisper.load_model(WHISPER_MODEL)
    except KeyboardInterrupt:
        if sys.platform == "darwin":  # macOS
            print("\n🚫 Loading cancelled by user (Ctrl+C).")
        else:
            print("\n🚫 Loading cancelled by user.")
        return None
    except Exception as e:
        print(f"\n❌ Error loading Whisper model: {e}")
        return None
    
    # Initialize worker pool
    try:
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
    except KeyboardInterrupt:
        if sys.platform == "darwin":  # macOS
            print("\n🚫 Initialization cancelled by user (Ctrl+C).")
        else:
            print("\n🚫 Initialization cancelled by user.")
        return None
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        return None
    
    # Threading control
    recording_event = threading.Event()
    recording_event.set()
    stop_event = threading.Event()
    cancelled = threading.Event()
    
    # Audio data collection
    audio_data = []
    
    # Keyboard handling setup
    fd, old_settings = setup_keyboard_handler()
    
    # Set up global signal handler (only works in main thread)
    try:
        _global_recording_state['recording_event'] = recording_event
        _global_recording_state['cancelled'] = cancelled
        _global_recording_state['original_handler'] = signal.signal(signal.SIGINT, _global_signal_handler)
        signal_handler_setup = True
    except ValueError as e:
        # Not in main thread - can't set up signal handler
        print("⚠️  Note: Ctrl+C handling not available (not in main thread)")
        signal_handler_setup = False
    
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
    
    def keyboard_listener():
        """Listen for keyboard input in a separate thread."""
        try:
            # Try advanced keyboard handling first
            if fd is not None:
                while recording_event.is_set() and not cancelled.is_set():
                    if check_keyboard_input(fd):
                        char = get_keyboard_char()
                        if char == '\r' or char == '\n':  # Enter key
                            print("\n✅ Recording stopped by user.")
                            recording_event.clear()
                            break
                        elif ord(char) == 3:  # Ctrl+C (ASCII 3)
                            print("\n🚫 Recording cancelled by user (Ctrl+C detected in keyboard listener).")
                            cancelled.set()
                            recording_event.clear()
                            break
                    threading.Event().wait(0.1)  # Small delay to prevent busy waiting
            else:
                # Fallback to simple input() for environments that don't support raw terminal
                try:
                    input()  # This will block until Enter is pressed
                    if recording_event.is_set():  # Only stop if still recording
                        print("\n✅ Recording stopped by user.")
                        recording_event.clear()
                except (EOFError, KeyboardInterrupt):
                    if recording_event.is_set():
                        print("\n🚫 Recording cancelled by user.")
                        cancelled.set()
                        recording_event.clear()
        except KeyboardInterrupt:
            # Backup Ctrl+C handling in keyboard listener
            print("\n🚫 Recording cancelled by user (Ctrl+C in keyboard thread).")
            cancelled.set()
            recording_event.clear()
        except Exception as e:
            logging.error(f"Keyboard listener error: {e}")
            # Final fallback - just wait for a bit and let other controls handle stopping
            pass
    
    try:
        # Start worker pool
        worker_pool.start()
        vad_service.start_recording()
        
        # Start chunk processing thread
        chunk_thread = threading.Thread(target=process_ready_chunks, daemon=True)
        chunk_thread.start()
        
        # Start keyboard listener thread
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        # Start audio recording with configured device (or override)
        device = device_override if device_override is not None else (None if DEFAULT_MIC_DEVICE == -1 else DEFAULT_MIC_DEVICE)
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, device=device)
        stream.start()
        
        # Wait for recording to be stopped (by Enter key, Ctrl+C, or other means)
        try:
            while recording_event.is_set() and not cancelled.is_set():
                threading.Event().wait(0.05)  # Check more frequently
        except KeyboardInterrupt:
            # Backup Ctrl+C handling in case signal handler doesn't work
            if sys.platform == "darwin":  # macOS
                print("\n🚫 Recording cancelled by user (Ctrl+C).")
            else:
                print("\n🚫 Recording cancelled by user.")
            cancelled.set()
            recording_event.clear()
        
        # Stop recording
        stream.stop()
        stream.close()
        
        # Check if recording was cancelled
        if cancelled.is_set():
            print("Recording cancelled. No file will be saved.")
            return None
        
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
        
        # Restore terminal settings
        restore_keyboard_handler(fd, old_settings)
        
        # Restore original signal handler
        if signal_handler_setup and _global_recording_state['original_handler']:
            signal.signal(signal.SIGINT, _global_recording_state['original_handler'])
            _global_recording_state['original_handler'] = None
            _global_recording_state['recording_event'] = None
            _global_recording_state['cancelled'] = None
        
        # Clean up components
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

def list_transcripts():
    """List all available transcripts with their IDs."""
    try:
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        transcripts = file_manager.list_transcripts()
        
        # Also check for legacy format files (only in SAVE_PATH)
        legacy_files = []
        if os.path.exists(SAVE_PATH):
            for filename in os.listdir(SAVE_PATH):
                if (filename.endswith(('.md', '.txt')) and 
                    TranscriptHeader.is_legacy_format_file(filename) and
                    not TranscriptHeader.is_id_format_file(filename)):
                    file_path = os.path.join(SAVE_PATH, filename)
                    try:
                        stat = os.stat(file_path)
                        mod_time = datetime.fromtimestamp(stat.st_mtime)
                        legacy_files.append((filename, mod_time))
                    except OSError:
                        continue
        
        # Sort legacy files by modification time (newest first)
        legacy_files.sort(key=lambda x: x[1], reverse=True)
        
        if not transcripts and not legacy_files:
            print("📝 No transcripts found.")
            return
        
        print("\n📋 Available Transcripts:")
        print("─" * 60)
        
        # Show new ID-format transcripts first
        if transcripts:
            print("🆔 New Format (ID-based):")
            for transcript_id, filename, creation_date in transcripts:
                date_str = creation_date.strftime("%Y-%m-%d %H:%M")
                print(f"   {filename} (ID: {transcript_id}, {date_str})")
            print()
        
        # Show legacy format transcripts
        if legacy_files:
            print("� Legacy Format (timestamp-based):")
            for filename, mod_time in legacy_files:
                date_str = mod_time.strftime("%Y-%m-%d %H:%M")
                print(f"   {filename} ({date_str})")
            print()
        
        if transcripts:
            print(f"💡 Use 'rec -XXXXXX' to reference ID-based transcripts")
        print(f"💡 New transcripts use format: descriptive-name_DDMMYYYY_000001.{OUTPUT_FORMAT}")
        
    except Exception as e:
        print(f"❌ Error listing transcripts: {e}")

def show_transcript(id_reference):
    """Show the content of a transcript by ID."""
    try:
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        content = file_manager.get_transcript_content(id_reference)
        
        if content:
            print(f"\n📄 Transcript {id_reference}:")
            print("─" * 50)
            print(content)
        else:
            print(f"❌ Transcript with ID '{id_reference}' not found.")
            print("💡 Use 'rec --list' to see available transcripts")
        
    except Exception as e:
        print(f"❌ Error showing transcript: {e}")

def append_to_transcript(id_reference):
    """Record new audio and append to existing transcript."""
    try:
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        
        # Check if transcript exists
        existing_path = file_manager.find_transcript(id_reference)
        if not existing_path:
            print(f"❌ Transcript with ID '{id_reference}' not found.")
            print("💡 Use 'rec --list' to see available transcripts")
            return
        
        clean_id = file_manager.id_generator.parse_reference_id(id_reference)
        print(f"🔗 Appending to transcript {clean_id}")
        
        # Show existing content preview
        existing_content = file_manager.get_transcript_content(id_reference)
        if existing_content:
            preview = existing_content[:200] + "..." if len(existing_content) > 200 else existing_content
            print(f"📄 Current content preview: {preview}")
        
        print("\n--- Recording additional content ---")
        
        # Record new audio
        new_transcript = record_audio_chunked()
        if not new_transcript:
            print("❌ No new content recorded.")
            return
        
        # Deduplicate the new content
        new_transcript = deduplicate_transcript(new_transcript)
        
        print("\n--- NEW CONTENT ---")
        print(new_transcript)
        print("--------------------")
        
        # Append to existing transcript
        updated_path = file_manager.append_to_transcript(id_reference, new_transcript)
        
        if updated_path:
            print(f"✅ Successfully appended to transcript {clean_id}")
            print(f"📁 Updated file: {updated_path}")
            
            # Copy combined content to clipboard if enabled
            if AUTO_COPY:
                combined_content = file_manager.get_transcript_content(id_reference)
                if combined_content:
                    pyperclip.copy(combined_content)
                    print("📋 Combined transcript copied to clipboard.")
        else:
            print(f"❌ Failed to append to transcript {clean_id}")
        
    except Exception as e:
        print(f"❌ Error appending to transcript: {e}")

def summarize_file(path_or_id):
    """Summarize and tag a file by path or transcript ID."""
    try:
        # Initialize summarization service
        summarizer = SummarizationService(
            ollama_model=OLLAMA_MODEL,
            ollama_timeout=OLLAMA_TIMEOUT,
            notes_folder=SAVE_PATH,  # Use same folder as transcripts for processed files
            max_content_length=OLLAMA_MAX_CONTENT_LENGTH
        )
        
        # Determine if input is a file path or transcript ID
        file_path = None
        
        if path_or_id.startswith('-') or path_or_id.isdigit():
            # It's a transcript ID reference
            file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
            file_path = file_manager.find_transcript(path_or_id)
            
            if not file_path:
                print(f"❌ Transcript with ID '{path_or_id}' not found.")
                print("💡 Use 'rec --list' to see available transcripts")
                return
            
            print(f"🔍 Found transcript: {os.path.basename(file_path)}")
        else:
            # It's a file path
            file_path = os.path.abspath(path_or_id)
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return
            
            # Check if it's a text file
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in ['.md', '.txt', '']:
                print(f"⚠️ File type '{ext}' may not be supported. Continuing anyway...")
        
        print(f"🤖 Summarizing file: {os.path.basename(file_path)}")
        
        # Check if this is a transcript file (don't copy to notes folder)
        is_transcript_file = file_path.startswith(SAVE_PATH)
        
        # Summarize the file
        success = summarizer.summarize_file(file_path, copy_to_notes=not is_transcript_file)
        
        if success:
            print("🎉 Summarization completed successfully!")
            
            # Copy to clipboard if enabled
            if AUTO_COPY and is_transcript_file:
                # For transcript files, copy the updated content
                file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
                # Extract ID from filename to get updated content
                import re
                id_match = re.search(r'_(\d+)\.(md|txt)$', file_path)
                if id_match:
                    transcript_id = id_match.group(1)
                    updated_content = file_manager.get_transcript_content(transcript_id)
                    if updated_content:
                        pyperclip.copy(updated_content)
                        print("📋 Updated transcript copied to clipboard.")
        else:
            print("❌ Summarization failed.")
        
    except Exception as e:
        print(f"❌ Error during summarization: {e}")

def main(args=None):
    try:
        # Set defaults if no args provided
        if args is None:
            args = type('Args', (), {})()
        
        # 1. Record Audio with real-time chunking and transcription
        device_override = args.device if hasattr(args, 'device') and args.device is not None else None
        transcribed_text = record_audio_chunked(device_override)
        if not transcribed_text:
            return
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C at any point in the main function
        if sys.platform == "darwin":  # macOS
            print("\n🚫 Operation cancelled by user (Ctrl+C).")
        else:
            print("\n🚫 Operation cancelled by user.")
        return
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
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

    # 4. Generate filename and save transcript
    file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
    
    # Try to get AI-generated filename if Ollama is available
    generated_filename = "transcript"  # default
    ollama_available = check_ollama_available()
    
    if ollama_available and AUTO_METADATA:
        metadata = get_combined_metadata_from_llm(transcribed_text)
        if metadata and metadata.get('filename'):
            generated_filename = metadata['filename']
    
    try:
        file_path, transcript_id = file_manager.create_new_transcript(
            transcribed_text, 
            generated_filename
        )
        print(f"✅ Successfully saved transcript to: {file_path}")
        print(f"🆔 Transcript ID: {transcript_id}")
    except Exception as e:
        print(f"❌ Error saving transcript: {e}")
        return

    # 5. Add AI-generated summary and tags to frontmatter (if enabled)
    final_path = file_path  # Use the ID-based file path
    
    if ollama_available and AUTO_METADATA:
        print("🤖 Generating summary and tags...")
        summarizer = SummarizationService(
            ollama_model=OLLAMA_MODEL,
            ollama_timeout=OLLAMA_TIMEOUT,
            max_content_length=OLLAMA_MAX_CONTENT_LENGTH
        )
        success = summarizer.summarize_file(file_path, copy_to_notes=False)
        if success:
            print("✅ Summary and tags added to transcript metadata")
        else:
            print("⚠️ Could not generate AI summary - transcript saved without metadata")
    elif not ollama_available:
        print("ℹ️  Ollama not available - transcript saved without AI metadata")

    # 6. Handle post-transcription actions
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
    parser.add_argument('id_reference', nargs='?', 
                       help='Reference existing transcript by ID (e.g., -123456)')
    parser.add_argument('--list', action='store_true',
                       help='List all transcripts with their IDs')
    parser.add_argument('--show', type=str, metavar='ID',
                       help='Show content of transcript by ID')
    parser.add_argument('--summarize', '--sum', type=str, metavar='PATH_OR_ID',
                       help='Summarize and tag a file by path or transcript ID (e.g., /path/to/file.md or -123)')
    
    # Set defaults to None so we can detect when they're not specified
    parser.set_defaults(copy=None, open=None, metadata=None)
    
    args = parser.parse_args()
    
    try:
        if not all([SAVE_PATH, OUTPUT_FORMAT, WHISPER_MODEL, OLLAMA_MODEL]):
            print("❌ Configuration is missing. Please run the setup.sh script first.")
        elif args.settings:
            settings_menu()
        elif args.list:
            list_transcripts()
        elif args.show:
            show_transcript(args.show)
        elif args.summarize:
            summarize_file(args.summarize)
        elif args.id_reference:
            append_to_transcript(args.id_reference)
        else:
            main(args)
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C at the script level
        if sys.platform == "darwin":  # macOS
            print("\n🚫 Script cancelled by user (Ctrl+C).")
        else:
            print("\n🚫 Script cancelled by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)