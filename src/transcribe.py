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
import time
from scipy.io.wavfile import write
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import shutil
import wave
from pathlib import Path
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

# Old load_prompts function removed - now using SummarizationService._load_prompts

def load_templates():
    """Load templates from templates.json file"""
    templates_path = os.path.join(os.path.dirname(__file__), '..', 'templates.json')
    try:
        with open(templates_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load templates file: {e}")
        return {}

# All AI functions moved to SummarizationService - transcribe.py just handles recording

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

# Old AI system removed - now using hierarchical SummarizationService only

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
            print("  6. 🔧 Command (Change command name)")
            print("  7. 🗑️  Uninstall (Remove aliases, venv, and config)")
            print("  8. 🚪 Exit")
            
            choice = input("\n👉 Choose a category (1-8): ").strip()
            
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
                command_settings()
            elif choice == "7":
                uninstall_settings()
            elif choice == "8":
                print("👋 Exiting settings...")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")
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

def command_settings():
    """Command settings submenu"""
    while True:
        # Read current command name
        current_command = os.getenv('COMMAND_NAME', 'rec')
        
        print(f"\n🔧 Command Settings")
        print("─" * 20)
        print(f"Current command: {current_command}")
        print(f"Usage: {current_command} (start recording)")
        print(f"Usage: {current_command} -s (settings)")
        print(f"Usage: {current_command} -l (list transcripts)")
        print(f"\n1. Change Command Name")
        print(f"2. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-2): ").strip()
        
        if choice == "1":
            print(f"\nCurrent command: {current_command}")
            print("Examples: rec, record, transcribe, voice, tr, etc.")
            print("Choose something that won't conflict with existing commands.")
            
            new_command = input("Enter new command name: ").strip()
            
            if new_command and new_command != current_command:
                # Update the .env file
                update_env_setting("COMMAND_NAME", new_command)
                
                # Update the alias in ~/.zshrc
                try:
                    # Get project directory and venv python path
                    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    venv_python = os.path.join(project_dir, 'venv', 'bin', 'python')
                    
                    # Remove old alias and add new one
                    if os.path.exists(os.path.expanduser("~/.zshrc")):
                        # Create backup
                        import shutil
                        shutil.copy(os.path.expanduser("~/.zshrc"), os.path.expanduser("~/.zshrc.backup"))
                        
                        # Remove old section and add new alias
                        with open(os.path.expanduser("~/.zshrc"), 'r') as f:
                            lines = f.readlines()
                        
                        # Find and remove the old section
                        new_lines = []
                        in_section = False
                        for line in lines:
                            if line.strip() == "# Added by Local Transcriber Setup":
                                in_section = True
                                continue
                            elif in_section and (line.strip() == "" or line.startswith("#") and not line.startswith("# Added by")):
                                in_section = False
                            
                            if not in_section:
                                new_lines.append(line)
                        
                        # Add new alias
                        new_lines.append("\n# Added by Local Transcriber Setup\n")
                        new_lines.append(f"alias {new_command}='{venv_python} {project_dir}/src/transcribe.py'\n")
                        
                        # Write back to file
                        with open(os.path.expanduser("~/.zshrc"), 'w') as f:
                            f.writelines(new_lines)
                        
                        print(f"✅ Command changed from '{current_command}' to '{new_command}'")
                        print(f"🔄 Please restart your terminal or run 'source ~/.zshrc' to use the new command")
                        print(f"💡 Your old command '{current_command}' will no longer work")
                        
                    else:
                        print("❌ Could not find ~/.zshrc file")
                        
                except Exception as e:
                    print(f"❌ Error updating alias: {e}")
                    print("💡 You may need to manually update your ~/.zshrc file")
            elif new_command == current_command:
                print("ℹ️  Command name is already set to that value")
            else:
                print("❌ Invalid command name")
        
        elif choice == "2":
            break
        else:
            print("❌ Invalid choice. Please select 1-2.")

def uninstall_settings():
    """Uninstall settings submenu"""
    while True:
        print(f"\n🗑️  Uninstall Settings")
        print("─" * 25)
        print("This will remove:")
        print("  • Shell aliases from ~/.zshrc")
        print("  • Python virtual environment (venv/)")
        print("  • Configuration file (.env)")
        print("  • Optionally remove transcripts")
        print(f"\n1. Run Uninstall")
        print(f"2. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-2): ").strip()
        
        if choice == "1":
            print("\n⚠️  This will completely remove the Local Transcriber installation.")
            confirm = input("Are you sure you want to continue? (y/N): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                # Get the project directory
                project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                uninstall_script = os.path.join(project_dir, 'uninstall.sh')
                
                if os.path.exists(uninstall_script):
                    print(f"🚀 Running uninstall script...")
                    try:
                        import subprocess
                        result = subprocess.run(['bash', uninstall_script], cwd=project_dir)
                        if result.returncode == 0:
                            print("✅ Uninstall completed successfully!")
                            print("👋 Thank you for using Local Transcriber!")
                            sys.exit(0)
                        else:
                            print("❌ Uninstall script failed")
                    except Exception as e:
                        print(f"❌ Error running uninstall script: {e}")
                else:
                    print(f"❌ Uninstall script not found at: {uninstall_script}")
                    print("💡 You can manually remove:")
                    print(f"  • Aliases from ~/.zshrc")
                    print(f"  • Virtual environment: {project_dir}/venv/")
                    print(f"  • Configuration: {project_dir}/.env")
            else:
                print("❌ Uninstall cancelled")
        
        elif choice == "2":
            break
        else:
            print("❌ Invalid choice. Please select 1-2.")

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

# Session management helper functions
def cleanup_session_file(audio_writer, session_file):
    """Clean up session file and writer"""
    if audio_writer:
        try:
            audio_writer.close()
        except:
            pass
    
    try:
        if session_file.exists():
            session_file.unlink()
    except:
        pass

def preserve_session_for_recovery(session_file, session_id, reason):
    """Preserve session file for recovery"""
    if not session_file.exists():
        return
    
    file_size = session_file.stat().st_size
    duration = file_size / (SAMPLE_RATE * 2) if file_size > 0 else 0
    
    print(f"💾 Audio session preserved: {session_file.name}")
    print(f"📊 Duration: {duration:.1f}s, Size: {file_size/1024/1024:.1f}MB")
    print(f"🔧 Reason: {reason}")
    print(f"🔄 Recover with: python transcribe.py --recover {session_id}")

def transcribe_session_file(session_file, whisper_model):
    """Transcribe a complete session file"""
    try:
        # Read WAV file directly using wave module
        with wave.open(str(session_file), 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            
        # Convert bytes to numpy array
        audio_data = np.frombuffer(frames, dtype=np.int16)
        
        # Convert to float32 normalized to [-1, 1]
        audio_data = audio_data.astype(np.float32) / 32767.0
        
        # Handle stereo to mono conversion if needed
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
        
        if len(audio_data) < 1600:  # Less than 0.1 seconds
            return None
        
        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != SAMPLE_RATE:
            # Simple resampling - for production use scipy.signal.resample
            ratio = SAMPLE_RATE / sample_rate
            new_length = int(len(audio_data) * ratio)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_length),
                np.arange(len(audio_data)),
                audio_data
            )
        
        if WHISPER_LANGUAGE and WHISPER_LANGUAGE.lower() != "auto":
            result = whisper_model.transcribe(audio_data, fp16=False, language=WHISPER_LANGUAGE)
        else:
            result = whisper_model.transcribe(audio_data, fp16=False)
        
        return result["text"]
        
    except Exception as e:
        print(f"❌ Session transcription failed: {e}")
        return None

def maintain_realtime_buffer(audio_buffer, max_seconds=30):
    """Keep realtime buffer under size limit"""
    max_samples = max_seconds * SAMPLE_RATE
    current_samples = sum(len(chunk) for chunk in audio_buffer)
    
    while current_samples > max_samples and len(audio_buffer) > 1:
        removed = audio_buffer.pop(0)
        current_samples -= len(removed)

def cleanup_services_with_timeout(worker_pool, vad_service, timeout=1.0):
    """Cleanup services with aggressive timeout protection"""
    try:
        def cleanup():
            try:
                if worker_pool:
                    worker_pool.stop()
            except:
                pass
            try:
                if vad_service:
                    vad_service.stop_recording()
            except:
                pass
        
        stop_thread = threading.Thread(target=cleanup)
        stop_thread.start()
        stop_thread.join(timeout=timeout)
        
        if stop_thread.is_alive():
            print("⚠️ Service cleanup timed out (continuing anyway)")
    except:
        pass

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
    except Exception:
        # Ensure we always set the cancelled flag even if printing fails
        if _global_recording_state['cancelled']:
            _global_recording_state['cancelled'].set()
        if _global_recording_state['recording_event']:
            _global_recording_state['recording_event'].clear()

def record_audio_chunked(device_override: Optional[int] = None) -> Optional[str]:
    """
    Records audio with progressive chunked transcription and graceful degradation.
    Three-tier fallback: real-time chunks -> chunk recovery -> full transcription.
    """
    # Create session-based temp file
    temp_audio_dir = Path(SAVE_PATH or tempfile.gettempdir()) / "audio_sessions"
    temp_audio_dir.mkdir(exist_ok=True)
    
    session_id = int(time.time())
    session_audio_file = temp_audio_dir / f"session_{session_id}.wav"
    
    # Platform-specific messaging for keyboard shortcuts
    if sys.platform == "darwin":  # macOS
        print("🔴 Recording... Press Enter to stop, Ctrl+C (^C) to cancel.")
    else:
        print("🔴 Recording... Press Enter to stop, Ctrl+C to cancel.")
    
    # Audio file writer for continuous saving
    audio_writer = None
    total_frames_written = 0
    
    # Chunk management for progressive transcription
    completed_chunks = {}  # chunk_id -> transcript
    failed_chunks = {}     # chunk_id -> audio_data (for later retry)
    chunk_order = []       # maintain order for assembly
    
    def initialize_audio_writer():
        nonlocal audio_writer
        try:
            audio_writer = wave.open(str(session_audio_file), 'wb')
            audio_writer.setnchannels(1)
            audio_writer.setsampwidth(2)
            audio_writer.setframerate(SAMPLE_RATE)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize audio session: {e}")
            return False
    
    if not initialize_audio_writer():
        return None
    
    # Initialize components
    chunker = AudioChunker(
        chunk_duration_seconds=CHUNK_DURATION_SECONDS,
        overlap_seconds=CHUNK_OVERLAP_SECONDS,
        sample_rate=SAMPLE_RATE
    )
    
    # Load Whisper model with graceful fallback
    whisper_model = None
    realtime_enabled = True
    
    try:
        whisper_model = whisper.load_model(WHISPER_MODEL)
    except KeyboardInterrupt:
        if sys.platform == "darwin":  # macOS
            print("\n🚫 Loading cancelled by user (Ctrl+C).")
        else:
            print("\n🚫 Loading cancelled by user.")
        cleanup_session_file(audio_writer, session_audio_file)
        return None
    except Exception as e:
        print(f"⚠️ Whisper model failed to load: {e}")
        print("⚠️ Continuing with audio-only recording (transcription will be done later)")
        realtime_enabled = False
    
    # Initialize worker pool with error handling and graceful degradation
    worker_pool = None
    vad_service = None
    
    if whisper_model and realtime_enabled:
        try:
            worker_pool = TranscriptionWorkerPool(
                whisper_model=whisper_model,
                whisper_language=WHISPER_LANGUAGE,
                num_workers=1,  # Use single worker to avoid race conditions
                max_retry_attempts=2  # Reduce retries to fail faster
            )
            
            vad_service = VADService(
                silence_threshold_chunks=SILENCE_TRIGGER_CHUNKS,
                chunk_duration_seconds=CHUNK_DURATION_SECONDS
            )
            
            worker_pool.start()
            vad_service.start_recording()
            
        except KeyboardInterrupt:
            if sys.platform == "darwin":  # macOS
                print("\n🚫 Initialization cancelled by user (Ctrl+C).")
            else:
                print("\n🚫 Initialization cancelled by user.")
            cleanup_session_file(audio_writer, session_audio_file)
            return None
        except Exception as e:
            print(f"⚠️ Real-time transcription setup failed: {e}")
            print("⚠️ Continuing with audio-only recording")
            realtime_enabled = False
            worker_pool = None
            vad_service = None
    
    # Threading control
    recording_event = threading.Event()
    recording_event.set()
    cancelled = threading.Event()
    audio_data_for_realtime = []
    
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
        """Stream audio to disk immediately + optional realtime processing"""
        nonlocal total_frames_written, realtime_enabled
        
        if recording_event.is_set() and audio_writer:
            try:
                # CRITICAL: Always write to disk first - this is our lossless guarantee
                audio_1d = indata.flatten()
                audio_16bit = (audio_1d * 32767).astype(np.int16)
                
                audio_writer.writeframes(audio_16bit.tobytes())
                audio_writer._file.flush()
                os.fsync(audio_writer._file.fileno())
                
                total_frames_written += len(audio_16bit)
                
                # Only do realtime processing if it's enabled and working
                if realtime_enabled:
                    try:
                        # Maintain small buffer for realtime processing
                        audio_data_for_realtime.append(audio_1d.copy())
                        maintain_realtime_buffer(audio_data_for_realtime)
                        
                        # Add to chunker for realtime feedback
                        chunker.add_audio_data(audio_1d.copy())
                    except Exception as e:
                        # If realtime processing fails, disable it but continue recording
                        print(f"⚠️ Realtime processing error (disabling): {e}")
                        realtime_enabled = False
                
            except Exception as e:
                print(f"⚠️ Audio write error: {e}")
    
    def process_ready_chunks():
        """Process chunks with error isolation and queuing"""
        nonlocal realtime_enabled
        
        if not realtime_enabled or not worker_pool:
            return
            
        while recording_event.is_set() and realtime_enabled:
            try:
                ready_chunks = chunker.get_ready_chunks()
                
                for chunk_info in ready_chunks:
                    if not cancelled.is_set() and realtime_enabled:
                        try:
                            # Handle different possible chunk formats from AudioChunker
                            chunk_data = None
                            chunk_id = None
                            timestamp = time.time()
                            
                            if isinstance(chunk_info, tuple):
                                # Flexible unpacking to handle different tuple sizes
                                if len(chunk_info) >= 1:
                                    chunk_data = chunk_info[0]
                                if len(chunk_info) >= 2:
                                    chunk_id = chunk_info[1]
                                if len(chunk_info) >= 3:
                                    timestamp = chunk_info[2]
                            else:
                                # Assume it's raw chunk data
                                chunk_data = chunk_info
                            
                            # Generate chunk_id if not provided
                            if chunk_id is None:
                                chunk_id = f"chunk_{len(chunk_order)}"
                            
                            if chunk_data is not None:
                                chunk_order.append(chunk_id)
                                
                                # Try immediate transcription
                                worker_pool.add_chunk((chunk_data, chunk_id, timestamp))
                                if vad_service:
                                    vad_service.analyze_chunk(chunk_data)
                            
                        except Exception as e:
                            print(f"⚠️ Chunk processing failed, disabling realtime: {e}")
                            # On any chunk processing error, disable realtime but continue recording
                            realtime_enabled = False
                            break
                            
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Chunk processing error, disabling realtime: {e}")
                # On any processing error, disable realtime but continue recording
                realtime_enabled = False
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
                            print("✅ Recording stopped by user.")
                            sys.stdout.flush()
                            recording_event.clear()
                            break
                        elif ord(char) == 3:  # Ctrl+C (ASCII 3)
                            print("🚫 Recording cancelled by user (Ctrl+C detected in keyboard listener).")
                            cancelled.set()
                            recording_event.clear()
                            break
                    threading.Event().wait(0.1)  # Small delay to prevent busy waiting
            else:
                # Fallback to simple input() for environments that don't support raw terminal
                try:
                    input()  # This will block until Enter is pressed
                    if recording_event.is_set():  # Only stop if still recording
                        print("✅ Recording stopped by user.")
                        sys.stdout.flush()
                        recording_event.clear()
                except (EOFError, KeyboardInterrupt):
                    if recording_event.is_set():
                        print("🚫 Recording cancelled by user.")
                        cancelled.set()
                        recording_event.clear()
        except KeyboardInterrupt:
            # Backup Ctrl+C handling in keyboard listener
            print("🚫 Recording cancelled by user (Ctrl+C in keyboard thread).")
            cancelled.set()
            recording_event.clear()
        except Exception as e:
            logging.error(f"Keyboard listener error: {e}")
            pass
    
    success = False
    transcript = None
    
    try:
        # Start processing threads (only if realtime is enabled)
        threads = []
        if realtime_enabled and worker_pool:
            chunk_thread = threading.Thread(target=process_ready_chunks, daemon=True)
            chunk_thread.start()
            threads.append(chunk_thread)
        
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()
        threads.append(keyboard_thread)
        
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
                print("🚫 Recording cancelled by user (Ctrl+C).")
            else:
                print("🚫 Recording cancelled by user.")
            cancelled.set()
            recording_event.clear()
        
        # Stop recording
        stream.stop()
        stream.close()
        
        # Close audio writer
        if audio_writer:
            audio_writer.close()
            audio_writer = None
        
        # Brief pause to ensure audio stream cleanup is complete
        time.sleep(0.1)
        
        # Check results
        if cancelled.is_set():
            print("🚫 Recording cancelled.")
            preserve_session_for_recovery(session_audio_file, session_id, "cancelled")
            return None
        
        if total_frames_written == 0:
            print("⚠️ No audio recorded")
            cleanup_session_file(None, session_audio_file)
            return None
        
        # Three-tier transcription strategy
        print()
        print("⏳ Finalizing transcription...")
        transcript = None
        
        # Strategy 1: Try to get real-time assembled transcript (with aggressive timeout)
        if realtime_enabled and worker_pool and chunk_order:
            print("🔄 Attempting to use real-time transcript...")
            try:
                # Give workers a brief moment to finish processing
                time.sleep(1.0)  # Reduced from 2.0 seconds
                
                # Try to get assembled transcript with very short timeout
                transcript_result = [None]
                transcript_error = [None]
                
                def get_transcript():
                    try:
                        transcript_result[0] = worker_pool.get_assembled_transcript()
                    except Exception as e:
                        transcript_error[0] = e
                
                transcript_thread = threading.Thread(target=get_transcript)
                transcript_thread.start()
                transcript_thread.join(timeout=2.0)  # Reduced from 5.0 seconds
                
                if transcript_thread.is_alive():
                    print("⚠️ Real-time transcript taking too long, using fallback")
                elif transcript_error[0]:
                    print(f"⚠️ Real-time transcript failed: {transcript_error[0]}")
                elif transcript_result[0] and transcript_result[0].strip():
                    transcript = transcript_result[0].strip()
                    print(f"✅ Real-time transcript ready ({len(chunk_order)} chunks processed)")
                else:
                    print("⚠️ Real-time transcript empty or incomplete")
                    
            except Exception as e:
                print(f"⚠️ Real-time transcript retrieval failed: {e}")
        
        # Strategy 2: Skip individual chunk recovery for now - go straight to full transcription
        
        # Strategy 3: Full file transcription (last resort)
        if not transcript:
            print("🔄 Using full file transcription...")
            
            # Load fresh model if needed
            if not whisper_model:
                try:
                    print("🎤 Loading Whisper model for transcription...")
                    whisper_model = whisper.load_model(WHISPER_MODEL)
                except Exception as e:
                    print(f"❌ Could not load Whisper model: {e}")
                    preserve_session_for_recovery(session_audio_file, session_id, "model_load_failed")
                    return None
            
            transcript = transcribe_session_file(session_audio_file, whisper_model)
        
        # Final result check
        if transcript and transcript.strip():
            success = True
            print("✅ Transcription completed successfully")
            return transcript.strip()
        else:
            print("⚠️ No speech detected in recording")
            preserve_session_for_recovery(session_audio_file, session_id, "no_speech")
            return None
        
    except Exception as e:
        print(f"❌ Recording error: {e}")
        preserve_session_for_recovery(session_audio_file, session_id, "error")
        return None
    
    finally:
        # Cleanup
        if audio_writer:
            try:
                audio_writer.close()
            except:
                pass
        
        # Restore terminal settings
        restore_keyboard_handler(fd, old_settings)
        
        # Restore original signal handler
        if signal_handler_setup and _global_recording_state.get('original_handler'):
            try:
                signal.signal(signal.SIGINT, _global_recording_state['original_handler'])
            except:
                pass
            _global_recording_state['original_handler'] = None
            _global_recording_state['recording_event'] = None
            _global_recording_state['cancelled'] = None
        
        # Stop services with aggressive timeout (only if they exist)
        if worker_pool or vad_service:
            cleanup_services_with_timeout(worker_pool, vad_service, timeout=1.0)
        
        # Clean up session file only on complete success
        if success and transcript:
            cleanup_session_file(None, session_audio_file)
            print("🗑️ Temporary audio file cleaned up")

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

def open_transcripts_folder():
    """Open the transcripts folder in Finder/Explorer."""
    try:
        if not SAVE_PATH or not os.path.exists(SAVE_PATH):
            print(f"❌ Transcripts folder not found: {SAVE_PATH}")
            print("💡 Run 'rec -s' to configure the save path")
            return
        
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", SAVE_PATH])
        elif sys.platform == "linux":  # Linux
            subprocess.run(["xdg-open", SAVE_PATH])
        elif sys.platform == "win32":  # Windows
            subprocess.run(["explorer", SAVE_PATH])
        else:
            print(f"📁 Transcripts folder: {SAVE_PATH}")
            return
        
        print(f"📂 Opened transcripts folder: {SAVE_PATH}")
        
    except Exception as e:
        print(f"❌ Error opening transcripts folder: {e}")
        print(f"📁 Transcripts folder location: {SAVE_PATH}")

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
        
    except ValueError as e:
        print(f"❌ {str(e)}")
        print("💡 Please resolve the naming conflict - multiple files have the same ID")
        print("💡 Use 'rec --list' to see all transcripts and their filenames")
    except Exception as e:
        print(f"❌ Error showing transcript: {e}")

def append_to_transcript(id_reference):
    """Record new audio and append to existing transcript."""
    try:
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        
        # Check if transcript exists
        try:
            existing_path = file_manager.find_transcript(id_reference)
        except ValueError as e:
            print(f"❌ {str(e)}")
            print("💡 Please resolve the naming conflict - multiple files have the same ID")
            print("💡 Use 'rec --list' to see all transcripts and their filenames")
            return
            
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
            
            try:
                file_path = file_manager.find_transcript(path_or_id)
            except ValueError as e:
                print(f"❌ {str(e)}")
                print("💡 Please resolve the naming conflict - multiple files have the same ID")
                print("💡 Use 'rec --list' to see all transcripts and their filenames")
                return
            
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

def list_recovery_sessions():
    """List available recovery sessions"""
    temp_audio_dir = Path(SAVE_PATH or tempfile.gettempdir()) / "audio_sessions"
    
    if not temp_audio_dir.exists():
        print("No recovery sessions available")
        return []
    
    session_files = list(temp_audio_dir.glob("session_*.wav"))
    
    if not session_files:
        print("No recovery sessions available")
        return []
    
    print(f"\n📋 Found {len(session_files)} recoverable sessions:")
    
    sessions = []
    for session_file in sorted(session_files):
        try:
            session_id = session_file.stem.split('_')[1]
            file_size = session_file.stat().st_size
            duration = file_size / (SAMPLE_RATE * 2)  # 16-bit mono
            timestamp = datetime.fromtimestamp(int(session_id))
            
            sessions.append({
                'id': session_id,
                'file': session_file,
                'duration': duration,
                'size_mb': file_size/1024/1024,
                'timestamp': timestamp
            })
            
            print(f"  {session_id}: {duration:.1f}s ({file_size/1024/1024:.1f}MB) - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"  ⚠️ Corrupted session: {session_file.name}")
    
    return sessions

def recover_session(session_id_or_latest=None):
    """Recover and transcribe a specific session"""
    sessions = list_recovery_sessions()
    
    if not sessions:
        return None
    
    # Find session
    if session_id_or_latest is None or session_id_or_latest == "latest":
        session = max(sessions, key=lambda s: s['timestamp'])
        print(f"\n🔄 Recovering latest session: {session['id']}")
    else:
        session = next((s for s in sessions if s['id'] == str(session_id_or_latest)), None)
        if not session:
            print(f"❌ Session {session_id_or_latest} not found")
            return None
    
    print(f"📁 Processing: {session['duration']:.1f}s recording from {session['timestamp'].strftime('%H:%M:%S')}")
    
    try:
        # Load Whisper model
        whisper_model = whisper.load_model(WHISPER_MODEL)
        
        # Transcribe session
        transcript = transcribe_session_file(session['file'], whisper_model)
        
        if transcript and transcript.strip():
            # Save transcript normally
            file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
            file_path, transcript_id = file_manager.create_new_transcript(transcript.strip(), "recovered_recording")
            
            print(f"✅ Recovery successful!")
            print(f"📄 Transcript {transcript_id} saved: {file_path}")
            
            # Add AI-generated summary and tags (if enabled)
            if AUTO_METADATA:
                print("🤖 Generating summary and tags...")
                summarizer = SummarizationService(
                    ollama_model=OLLAMA_MODEL,
                    ollama_timeout=OLLAMA_TIMEOUT,
                    max_content_length=OLLAMA_MAX_CONTENT_LENGTH
                )
                if summarizer.check_ollama_available():
                    success = summarizer.summarize_file(file_path, copy_to_notes=False)
                    if success:
                        print("✅ Summary and tags added to transcript metadata")
                    else:
                        print("⚠️ Could not generate AI summary - transcript saved without metadata")
                else:
                    print("ℹ️  Ollama not available - transcript saved without AI metadata")
            
            # Clean up session file after successful recovery
            session['file'].unlink()
            print(f"🗑️ Session file cleaned up")
            
            return transcript.strip()
        else:
            print("⚠️ No speech detected in recovered session")
            return None
            
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        return None

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

    # 3. Copy to clipboard immediately (before LLM processing)
    if AUTO_COPY:
        pyperclip.copy(transcribed_text)
        print()
        print("📋 Transcription copied to clipboard.")

    # 4. Save transcript with default filename first, then let AI rename it
    file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
    
    try:
        file_path, transcript_id = file_manager.create_new_transcript(
            transcribed_text, 
            "transcript"  # Use default name initially
        )
        print(f"✅ Transcript {transcript_id} successfully saved")
    except Exception as e:
        print(f"❌ Error saving transcript: {e}")
        return

    # 5. Add AI-generated summary, tags, and proper filename (if enabled)
    if AUTO_METADATA:
        print("🤖 Generating summary and tags...")
        summarizer = SummarizationService(
            ollama_model=OLLAMA_MODEL,
            ollama_timeout=OLLAMA_TIMEOUT,
            max_content_length=OLLAMA_MAX_CONTENT_LENGTH
        )
        if summarizer.check_ollama_available():
            success = summarizer.summarize_file(file_path, copy_to_notes=False)
            if success:
                print("✅ Summary and tags added to transcript metadata")
            else:
                print("⚠️ Could not generate AI summary - transcript saved without metadata")
        else:
            print("ℹ️  Ollama not available - transcript saved without AI metadata")

    # 6. Handle post-transcription actions
    summarizer_for_check = SummarizationService(ollama_model=OLLAMA_MODEL)
    handle_post_transcription_actions(transcribed_text, file_path, summarizer_for_check.check_ollama_available(), args)

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
    parser.add_argument('-l', '--list', action='store_true',
                       help='List all transcripts with their IDs')
    parser.add_argument('-v', '--view', type=str, metavar='ID', dest='show',
                       help='Show content of transcript by ID')
    parser.add_argument('-g', '--genai', type=str, metavar='PATH_OR_ID', dest='summarize',
                       help='AI analysis and tagging of a file by path or transcript ID (e.g., /path/to/file.md or -123)')
    parser.add_argument('-o', '--open-folder', action='store_true',
                       help='Open the transcripts folder in Finder/Explorer')
    parser.add_argument('-r', '--recover', nargs='?', const='latest', 
                       help='Recover session by ID or "latest"')
    parser.add_argument('-ls', '--list-sessions', action='store_true', 
                       help='List recoverable sessions')
    
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
        elif args.open_folder:
            open_transcripts_folder()
        elif args.list_sessions:
            list_recovery_sessions()
        elif args.recover:
            recover_session(args.recover)
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