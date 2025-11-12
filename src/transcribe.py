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
import argparse
import threading
import queue
import logging
import uuid
import signal
import select
import termios
import tty
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

# Import our new ID-based transcript management
from transcript_manager import TranscriptFileManager, AudioFileManager
from id_generator import TranscriptIDGenerator
from file_header import TranscriptHeader

# Import summarization service
from summarization_service import SummarizationService

# Import streaming transcription components
from audio_buffer import CircularAudioBuffer
from volume_segmenter import VolumeSegmenter, SegmentProcessor
from quick_transcript import QuickTranscriptAssembler
from background_enhancer import BackgroundEnhancer
from loading_indicator import LoadingIndicator
from safety_net import SafetyNetManager, SafetyNetIntegrator

# Import VAD service for auto-stop
from vad_service import VADService

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
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))  # Default 3 minutes for local LLMs
SAMPLE_RATE = 16000 # 16kHz is standard for Whisper

# Streaming transcription configuration (now the default)
STREAMING_BUFFER_SIZE_SECONDS = int(os.getenv("STREAMING_BUFFER_SIZE_SECONDS", "300"))  # 5 minutes
STREAMING_MIN_SEGMENT_DURATION = int(os.getenv("STREAMING_MIN_SEGMENT_DURATION", "30"))  # 30 seconds
STREAMING_TARGET_SEGMENT_DURATION = int(os.getenv("STREAMING_TARGET_SEGMENT_DURATION", "60"))  # 60 seconds
STREAMING_MAX_SEGMENT_DURATION = int(os.getenv("STREAMING_MAX_SEGMENT_DURATION", "90"))  # 90 seconds
STREAMING_VERBOSE = os.getenv("STREAMING_VERBOSE", "false").lower() == "true"

# Legacy configuration (for compatibility)
SILENCE_DURATION_SECONDS = int(os.getenv("SILENCE_DURATION_SECONDS", "120"))

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
    
    # Strip whitespace from value to prevent spacing issues
    value = str(value).strip()
    
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
        current_api_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
        
        timeout_minutes = current_timeout // 60
        timeout_seconds = current_timeout % 60
        timeout_str = f"{timeout_minutes}m {timeout_seconds}s" if timeout_minutes > 0 else f"{timeout_seconds}s"
        
        print(f"\n🤖 AI Settings")
        print("─" * 15)
        print(f"Current Ollama Model: {current_model}")
        print(f"Ollama API URL: {current_api_url}")
        print(f"Auto Metadata: {'Yes' if current_metadata else 'No'}")
        print(f"Ollama Timeout: {timeout_str}")
        print(f"Max Content Length: {current_max_length:,} characters")
        print(f"\n1. Change Ollama Model")
        print(f"2. Change Ollama API URL")
        print(f"3. Toggle Auto Metadata")
        print(f"4. Change Ollama Timeout")
        print(f"5. Change Max Content Length")
        print(f"6. ← Back to Main Menu")
        
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
            new_url = input("\nEnter Ollama API URL (e.g., http://localhost:11434/api/generate): ").strip()
            if new_url:
                update_env_setting("OLLAMA_API_URL", new_url)
                print(f"✅ Ollama API URL changed to: {new_url}")
                print("⚠️ Restart the script to use the new URL")

        elif choice == "3":
            new_setting = input("Auto generate AI metadata? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_METADATA", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto metadata changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "4":
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
        
        elif choice == "5":
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
        
        elif choice == "6":
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

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
        # Read current streaming settings (streaming is now always active)
        current_verbose = os.getenv('STREAMING_VERBOSE', 'false').lower() == 'true'
        current_buffer = int(os.getenv('STREAMING_BUFFER_SIZE_SECONDS', '300'))
        current_min = int(os.getenv('STREAMING_MIN_SEGMENT_DURATION', '30'))
        current_target = int(os.getenv('STREAMING_TARGET_SEGMENT_DURATION', '60'))
        current_max = int(os.getenv('STREAMING_MAX_SEGMENT_DURATION', '90'))
        
        print(f"\n⚡ Streaming Performance Settings")
        print("─" * 35)
        print(f"Mode: Streaming (active)")
        print(f"No Speech Detection: {os.getenv('SILENCE_DURATION_SECONDS', '120')} seconds")
        print(f"Streaming Buffer: {current_buffer}s ({current_buffer//60}m {current_buffer%60}s)")
        print(f"Streaming Segments: {current_min}s-{current_target}s-{current_max}s (min-target-max)")
        print(f"Streaming Verbose: {'Yes' if current_verbose else 'No'}")
        print(f"\n1. Change No Speech Detection Duration")
        print(f"2. Configure Streaming Buffer Size")
        print(f"3. Configure Streaming Segment Durations")
        print(f"4. Toggle Streaming Verbose Mode")
        print(f"5. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-5): ").strip()
        
        if choice == "1":
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
        
        elif choice == "2":
            print(f"\nCurrent buffer size: {current_buffer} seconds ({current_buffer//60}m {current_buffer%60}s)")
            print("Recommended buffer sizes:")
            print("  • 180s (3m)  - Short sessions, low memory")
            print("  • 300s (5m)  - Balanced (default)")
            print("  • 600s (10m) - Long sessions, high quality")
            
            new_buffer = input(f"Enter buffer size in seconds (60-1200) [current: {current_buffer}]: ").strip()
            try:
                buffer = int(new_buffer) if new_buffer else current_buffer
                if 60 <= buffer <= 1200:
                    minutes = buffer // 60
                    seconds = buffer % 60
                    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                    update_env_setting("STREAMING_BUFFER_SIZE_SECONDS", str(buffer))
                    print(f"✅ Streaming buffer size changed to: {buffer} seconds ({time_str})")
                    print("⚠️ Restart the script to use the new setting")
                else:
                    print("❌ Buffer size must be between 60 and 1200 seconds")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "3":
            print(f"\nCurrent segment durations: {current_min}s-{current_target}s-{current_max}s")
            print("Segment duration rules:")
            print("  • Min: Shortest allowed segment (avoid noise)")
            print("  • Target: Preferred segment length (optimal for Whisper)")
            print("  • Max: Force break point (prevent memory issues)")
            
            new_min = input(f"Enter minimum duration (10-60s) [current: {current_min}]: ").strip()
            new_target = input(f"Enter target duration (30-120s) [current: {current_target}]: ").strip()
            new_max = input(f"Enter maximum duration (60-180s) [current: {current_max}]: ").strip()
            
            try:
                min_dur = int(new_min) if new_min else current_min
                target_dur = int(new_target) if new_target else current_target
                max_dur = int(new_max) if new_max else current_max
                
                if (10 <= min_dur <= 60 and 30 <= target_dur <= 120 and 60 <= max_dur <= 180 and
                    min_dur <= target_dur <= max_dur):
                    update_env_setting("STREAMING_MIN_SEGMENT_DURATION", str(min_dur))
                    update_env_setting("STREAMING_TARGET_SEGMENT_DURATION", str(target_dur))
                    update_env_setting("STREAMING_MAX_SEGMENT_DURATION", str(max_dur))
                    print(f"✅ Segment durations changed to: {min_dur}s-{target_dur}s-{max_dur}s")
                    print("⚠️ Restart the script to use the new settings")
                else:
                    print("❌ Invalid durations. Ensure: 10≤min≤60, 30≤target≤120, 60≤max≤180, min≤target≤max")
            except ValueError:
                print("❌ Please enter valid numbers")
        
        elif choice == "4":
            new_verbose = input("Enable streaming verbose mode? (y/n): ").lower()
            if new_verbose in ['y', 'n']:
                update_env_setting("STREAMING_VERBOSE", 'true' if new_verbose == 'y' else 'false')
                print(f"✅ Streaming verbose mode changed to: {'Yes' if new_verbose == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "5":
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")

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

def record_audio_streaming(device_override: Optional[int] = None, verbose: bool = False) -> Tuple[Optional[str], Optional[Path], Optional[str], Optional[str]]:
    """
    Records audio using the new streaming transcription system.
    Simplified version using sounddevice for compatibility.
    
    Returns:
        Tuple[Optional[str], Optional[Path]]: (transcribed_text, master_audio_file_path)
    """
    # Suppress all logging unless verbose mode is enabled
    if not verbose:
        import logging
        logging.getLogger('audio_buffer').setLevel(logging.ERROR)
        logging.getLogger('volume_segmenter').setLevel(logging.ERROR)
        logging.getLogger('safety_net').setLevel(logging.ERROR)
        logging.getLogger('quick_transcript').setLevel(logging.ERROR)
        logging.getLogger('background_enhancer').setLevel(logging.ERROR)
        logging.getLogger('audio_manager').setLevel(logging.ERROR)
        import warnings
        warnings.filterwarnings("ignore")
    
    # Generate session ID
    session_id = f"stream_{int(time.time())}"
    
    # Create master audio file for safety net
    temp_audio_dir = Path(SAVE_PATH or tempfile.gettempdir()) / "audio_sessions"
    temp_audio_dir.mkdir(exist_ok=True)
    master_audio_file = temp_audio_dir / f"{session_id}.wav"
    
    # Initialize components
    try:
        # Audio buffer (5-minute rolling buffer)
        audio_buffer = CircularAudioBuffer(
            capacity_seconds=STREAMING_BUFFER_SIZE_SECONDS,
            sample_rate=SAMPLE_RATE,
            channels=1
        )
        
        # Volume segmenter with configurable durations
        from volume_segmenter import VolumeConfig
        volume_config = VolumeConfig(
            min_segment_duration=STREAMING_MIN_SEGMENT_DURATION,
            target_segment_duration=STREAMING_TARGET_SEGMENT_DURATION,
            max_segment_duration=STREAMING_MAX_SEGMENT_DURATION
        )
        volume_segmenter = VolumeSegmenter(
            audio_buffer=audio_buffer,
            config=volume_config,
            verbose=verbose
        )
        
        # Initialize audio segment processor
        segment_extractor = SegmentProcessor(audio_buffer)
        
        # Determine audio device
        device = device_override if device_override is not None else (None if DEFAULT_MIC_DEVICE == -1 else DEFAULT_MIC_DEVICE)
        
        # Initialize safety net
        safety_net = SafetyNetManager(
            safety_log_path=os.path.join(SAVE_PATH or ".", "safety_log.json"),
            auto_recovery=True,
            max_retry_attempts=3
        )
        
        # Start safety net session
        safety_record = safety_net.start_session(session_id, str(master_audio_file))
        streaming_attempt = safety_net.register_processing_attempt(
            session_id, "streaming", {"buffer_size": STREAMING_BUFFER_SIZE_SECONDS}
        )
        
        # Quick transcript assembler
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        audio_manager = AudioFileManager(SAVE_PATH)
        assembler = QuickTranscriptAssembler(
            transcript_manager=file_manager,
            audio_manager=audio_manager,
            auto_clipboard=AUTO_COPY
        )
        
        # Background enhancer for quality improvement
        summarizer = SummarizationService(
            ollama_model=OLLAMA_MODEL,
            ollama_api_url=OLLAMA_API_URL,
            ollama_timeout=OLLAMA_TIMEOUT,
            max_content_length=OLLAMA_MAX_CONTENT_LENGTH
        )
        def enhancement_completed(task):
            """Callback when background enhancement completes."""
            print(f"🎯 Background complete: enhanced transcript saved, audio cleaned up")
        
        enhancer = BackgroundEnhancer(
            transcript_manager=file_manager,
            audio_manager=audio_manager,
            summarization_service=summarizer,
            auto_cleanup=True
        )
        enhancer.task_completed_callback = enhancement_completed
        
        # Initialize components without logging details
        
    except Exception as e:
        print(f"❌ Failed to initialize streaming components: {e}")
        return None, None
    
    # Audio file writer for master file
    audio_writer = None
    recording_active = threading.Event()
    recording_active.set()
    user_stopped = threading.Event()
    
    def initialize_audio_writer():
        nonlocal audio_writer
        try:
            audio_writer = wave.open(str(master_audio_file), 'wb')
            audio_writer.setnchannels(1)
            audio_writer.setsampwidth(2)
            audio_writer.setframerate(SAMPLE_RATE)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize master audio file: {e}")
            return False
    
    if not initialize_audio_writer():
        return None, None
    
    def keyboard_listener():
        """Listen for user input to stop recording."""
        try:
            # Use a simpler approach that works better with audio callbacks
            while recording_active.is_set():
                try:
                    # Check for input without blocking indefinitely
                    import select
                    import sys
                    
                    if sys.stdin in select.select([sys.stdin], [], [], 0.5)[0]:
                        line = input()
                        if recording_active.is_set():
                            print("\n✅ Recording stopped by user.")
                            user_stopped.set()
                            recording_active.clear()
                            break
                except (select.error, OSError):
                    # Fallback for systems where select doesn't work
                    try:
                        input()
                        if recording_active.is_set():
                            print("\n✅ Recording stopped by user.")
                            user_stopped.set()
                            recording_active.clear()
                            break
                    except (EOFError, KeyboardInterrupt):
                        if recording_active.is_set():
                            print("\n🚫 Recording cancelled by user.")
                            user_stopped.set()
                            recording_active.clear()
                            break
                time.sleep(0.1)
        except (EOFError, KeyboardInterrupt):
            if recording_active.is_set():
                print("\n🚫 Recording cancelled by user.")
                user_stopped.set()
                recording_active.clear()
    
    def audio_callback(indata, frames, time_info, status):
        """Process incoming audio data."""
        if not recording_active.is_set():
            return
            
        try:
            # Convert to flat mono array
            audio_data = indata.flatten()
            
            # Write to master file immediately
            audio_16bit = (audio_data * 32767).astype(np.int16)
            audio_writer.writeframes(audio_16bit.tobytes())
            audio_writer._file.flush()
            
            # Add to circular buffer for streaming
            audio_buffer.write(audio_data)
            
            # Optional verbose monitoring
            if verbose and hasattr(audio_callback, 'call_count'):
                audio_callback.call_count += 1
                if audio_callback.call_count % 100 == 0:  # Every ~2 seconds
                    duration = audio_callback.call_count * frames / SAMPLE_RATE
                    print(f"🎙️  Recording: {duration:.1f}s", end='\r', flush=True)
            elif verbose:
                audio_callback.call_count = 1
                
        except Exception as e:
            if verbose:
                print(f"⚠️ Audio callback error: {e}")
    
    try:
        # Start keyboard listener
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        # Platform-specific instructions
        if sys.platform == "darwin":  # macOS
            print("🔴 Recording... Press Enter to stop, or Ctrl+C (^C) to cancel.")
        else:
            print("🔴 Recording... Press Enter to stop, or Ctrl+C to cancel.")
        
        print("💡 If Enter doesn't work, use Ctrl+C to stop recording safely.")
        
        # Start recording with sounddevice
        audio_buffer.start_recording()
        volume_segmenter.start_analysis()
        assembler.start_session(session_id)
        
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            callback=audio_callback,
            device=device,
            blocksize=1024
        )
        stream.start()
        
        # Monitor recording with live feedback
        start_time = time.time()
        last_status_time = time.time()
        segments_processed = 0
        
        while recording_active.is_set():
            current_time = time.time()
            duration = current_time - start_time
            
            # Process completed segments
            try:
                # First, analyze audio to detect new segments
                new_segments = volume_segmenter.analyze_and_segment()
                if verbose and new_segments:
                    print(f"\n🔍 Detected {len(new_segments)} new segments")
                
                # Get all detected segments
                completed_segments = volume_segmenter.get_detected_segments()
                        
                for segment_info in completed_segments[segments_processed:]:
                    try:
                        # Extract audio data for this segment
                        audio_data = segment_extractor.extract_segment_audio(segment_info)
                        if audio_data is not None:
                            assembler.add_segment_for_transcription(segment_info, audio_data)
                            segments_processed += 1
                            if verbose:
                                print(f"\n📦 Processed segment {segments_processed}: {segment_info.duration:.1f}s")
                        else:
                            if verbose:
                                print(f"⚠️ Could not extract audio for segment {segments_processed}")
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Segment processing error: {e}")
            except Exception as e:
                if verbose:
                    print(f"⚠️ Segmentation error: {e}")
            
            # Sleep briefly to prevent excessive CPU usage
            time.sleep(0.1)
            
            time.sleep(0.5)  # Check frequently for responsiveness
        
        # Stop recording
        stream.stop()
        stream.close()

        # Close audio writer
        if audio_writer:
            audio_writer.close()
            audio_writer = None

        audio_buffer.stop_recording()
        
        # Start loading indicator for processing
        loader = LoadingIndicator("🎤 Processing audio...")
        loader.start()
        
        try:
            # Process any final segments
            loader.update("🔍 Analyzing segments...")
            
            try:
                # Force flush any remaining segment
                final_segment = volume_segmenter.flush_remaining_segment()

                # Get any remaining segments
                final_segments = volume_segmenter.get_detected_segments()
                        
                for segment_info in final_segments[segments_processed:]:
                    try:
                        # Extract audio data for this segment
                        audio_data = segment_extractor.extract_segment_audio(segment_info)
                        if audio_data is not None:
                            assembler.add_segment_for_transcription(segment_info, audio_data)
                            segments_processed += 1
                            if verbose:
                                print(f"📦 Final segment: {segment_info.duration:.1f}s")
                        else:
                            if verbose:
                                print(f"⚠️ Could not extract final segment audio")
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Final segment error: {e}")
            except Exception as e:
                if verbose:
                    print(f"⚠️ Final processing error: {e}")
            
            # Finalize quick transcript
            loader.update("📝 Finalizing transcript...")
            quick_transcript = assembler.finalize_transcript()
            
            # Check if we have real transcription content (not just placeholder)
            has_real_content = bool(quick_transcript and quick_transcript.has_content)
            
            if has_real_content:
                # Mark streaming attempt as successful
                safety_net.complete_processing_attempt(
                    session_id, streaming_attempt, 
                    output_files=[quick_transcript.file_path] if quick_transcript.file_path else [],
                    success=True
                )
                
                # Start background enhancement
                enhancement_attempt = safety_net.register_processing_attempt(
                    session_id, "enhanced", {"master_audio": str(master_audio_file)}
                )
                
                enhancer.queue_enhancement(quick_transcript, str(master_audio_file))
                
                # Complete safety net session
                safety_net.complete_session(session_id, success=True, 
                                           final_output_files=[str(master_audio_file)])
                
                loader.stop()  # Clear loading indicator
                return quick_transcript.transcript_text.strip(), master_audio_file, quick_transcript.file_path, quick_transcript.transcript_id
                
            else:
                # Streaming failed - fall back to full-file transcription
                loader.update("🔄 Running fallback transcription...")
                
                # Mark streaming attempt as failed
                safety_net.complete_processing_attempt(
                    session_id, streaming_attempt, success=False,
                    error_message="No segments detected"
                )
                
                # Try full-file Whisper transcription as fallback
                try:
                    import whisper
                    whisper_model = whisper.load_model(WHISPER_MODEL)
                    result = whisper_model.transcribe(str(master_audio_file), language=WHISPER_LANGUAGE)
                    fallback_text = result['text'].strip()
                    
                    if fallback_text:
                        
                        # Mark fallback as successful
                        fallback_attempt = safety_net.register_processing_attempt(
                            session_id, "fallback", {"audio_file": str(master_audio_file)}
                        )
                        safety_net.complete_processing_attempt(
                            session_id, fallback_attempt, success=True,
                            output_files=[str(master_audio_file)]
                        )
                        safety_net.complete_session(session_id, success=True, 
                                                   final_output_files=[str(master_audio_file)])
                        
                        loader.stop()  # Clear loading indicator
                        return fallback_text, master_audio_file, None, None
                    
                except Exception as fallback_error:
                    pass
                
                # Both streaming and fallback failed
                safety_net.complete_session(session_id, success=False)
                loader.stop()  # Clear loading indicator
                return None, None, None, None
        
        except Exception as e:
            loader.stop()  # Clear loading indicator on error
            raise
            
    except KeyboardInterrupt:
        return None, None, None, None
        
    except Exception as e:
        safety_net.complete_processing_attempt(
            session_id, streaming_attempt, success=False,
            error_message=str(e)
        )
        safety_net.complete_session(session_id, success=False)
        return None, None
    
    finally:
        # Cleanup
        try:
            if audio_writer:
                audio_writer.close()
        except:
            pass
        try:
            volume_segmenter.stop_analysis()
        except:
            pass


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
        transcripts = file_manager.list_transcripts_with_audio()
        
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
            print(f"   {'ID':<6} {'Created':<16} {'Audio':<8} {'Duration':<10} {'Filename'}")
            print("   " + "─" * 80)
            
            for transcript_id, filename, creation_date, audio_count, total_duration in transcripts:
                date_str = creation_date.strftime("%Y-%m-%d %H:%M")
                audio_str = f"{audio_count}" if audio_count > 0 else "-"
                duration_str = f"{total_duration:.1f}s" if audio_count > 0 else "-"
                print(f"   {transcript_id:<6} {date_str:<16} {audio_str:<8} {duration_str:<10} {filename}")
            
        
        # Show legacy format transcripts
        if legacy_files:
            print("� Legacy Format (timestamp-based):")
            for filename, mod_time in legacy_files:
                date_str = mod_time.strftime("%Y-%m-%d %H:%M")
                print(f"   {filename} ({date_str})")
            
        
        if transcripts:
            print(f"\n💡 Use 'rec -XXXXXX' to reference ID-based transcripts")
            print(f"💡 Use 'rec --audio XXXXXX' to see audio files for a transcript")
            print(f"💡 Use 'rec --reprocess XXXXXX' to reprocess all audio for a transcript")
        print(f"💡 New transcripts use format: XXXXXX_DDMMYYYY_descriptive-name.{OUTPUT_FORMAT}")
        print(f"💡 Use 'rec --reprocess-failed' to process orphaned audio files")
        
    except Exception as e:
        print(f"❌ Error listing transcripts: {e}")

def transcribe_audio_file(audio_path: str) -> str:
    """Transcribe a single audio file using the same method as the main transcription."""
    from pathlib import Path
    import whisper
    
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load Whisper model
        whisper_model = whisper.load_model(WHISPER_MODEL)
        
        # Transcribe the audio file
        result = whisper_model.transcribe(str(audio_file), verbose=False)
        transcript_text = result.get('text', '').strip()
        
        return transcript_text if transcript_text else ""
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

def reprocess_transcript_command(id_reference: str, overwrite_existing: bool = False):
    """Reprocess all audio files for a specific transcript ID."""
    try:
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        
        # Check if transcript ID has audio files
        audio_files_info = file_manager.get_audio_files_for_transcript(id_reference)
        if not audio_files_info:
            print(f"❌ No audio files found for transcript ID {id_reference}")
            return
        
        print(f"🎵 Found {len(audio_files_info)} audio files for transcript {id_reference}:")
        for audio_info in audio_files_info:
            filename = os.path.basename(audio_info['path'])
            duration_str = f"{audio_info['duration']:.1f}s" if audio_info['exists'] else "Missing"
            status = "✅" if audio_info['exists'] else "❌"
            print(f"   {status} {filename} ({duration_str})")
        
        # Ask for confirmation
        response = input(f"\n🔄 Reprocess {len(audio_files_info)} audio files? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Reprocessing cancelled")
            return
        
        # Define summarization callback if AI is enabled
        summarization_callback = None
        if AUTO_METADATA:
            try:
                summarizer = SummarizationService(
                    ollama_model=OLLAMA_MODEL,
                    ollama_api_url=OLLAMA_API_URL,
                    ollama_timeout=OLLAMA_TIMEOUT,
                    max_content_length=OLLAMA_MAX_CONTENT_LENGTH
                )
                if summarizer.check_ollama_available():
                    def summarize_transcript(transcript_text: str) -> dict:
                        """Generate AI summary and metadata."""
                        return summarizer.get_metadata(transcript_text) or {}
                    
                    summarization_callback = summarize_transcript
                else:
                    print("⚠️ Ollama not available - skipping AI summarization")
                    
            except Exception as e:
                print(f"⚠️ AI summarization not available: {str(e)}")
        
        # Perform reprocessing
        success, transcript_path, processed_files = file_manager.reprocess_transcript_audio(
            id_reference,
            transcription_callback=transcribe_audio_file,
            summarization_callback=summarization_callback,
            overwrite_existing=overwrite_existing
        )
        
        if success:
            print(f"\n✅ Reprocessing completed successfully!")
            print(f"📁 Transcript: {os.path.basename(transcript_path)}")
            print(f"🎵 Processed {len(processed_files)} audio files:")
            for filename in processed_files:
                print(f"   - {filename}")
        else:
            print(f"\n❌ Reprocessing failed: {transcript_path}")
    
    except Exception as e:
        print(f"❌ Error during reprocessing: {e}")

def reprocess_failed_command():
    """Reprocess all orphaned audio files."""
    try:
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        
        # Define summarization callback if AI is enabled
        summarization_callback = None
        if AUTO_METADATA:
            try:
                summarizer = SummarizationService(
                    ollama_model=OLLAMA_MODEL,
                    ollama_api_url=OLLAMA_API_URL,
                    ollama_timeout=OLLAMA_TIMEOUT,
                    max_content_length=OLLAMA_MAX_CONTENT_LENGTH
                )
                if summarizer.check_ollama_available():
                    def summarize_transcript(transcript_text: str) -> dict:
                        """Generate AI summary and metadata."""
                        return summarizer.get_metadata(transcript_text) or {}
                    
                    summarization_callback = summarize_transcript
                else:
                    print("⚠️ Ollama not available - skipping AI summarization")
                    
            except Exception as e:
                print(f"⚠️ AI summarization not available: {str(e)}")
        
        # Perform batch reprocessing
        results = file_manager.reprocess_all_failed_transcripts(
            transcription_callback=transcribe_audio_file,
            summarization_callback=summarization_callback
        )
        
        if results:
            successful = [r for r in results if r[1]]
            failed = [r for r in results if not r[1]]
            
            print(f"\n📊 Batch reprocessing completed:")
            print(f"✅ Successful: {len(successful)}")
            print(f"❌ Failed: {len(failed)}")
            
            if successful:
                print(f"\n✅ Successfully reprocessed:")
                for transcript_id, _, message in successful:
                    print(f"   - ID {transcript_id}: {message}")
            
            if failed:
                print(f"\n❌ Failed to reprocess:")
                for transcript_id, _, message in failed:
                    print(f"   - ID {transcript_id}: {message}")
        else:
            print("✅ No orphaned audio files found to reprocess")
    
    except Exception as e:
        print(f"❌ Error during batch reprocessing: {e}")

def show_audio_files(id_reference):
    """Show audio files associated with a transcript."""
    try:
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        
        # Check if transcript exists
        try:
            existing_path = file_manager.find_transcript(id_reference)
        except ValueError as e:
            print(f"❌ {str(e)}")
            return
            
        if not existing_path:
            print(f"❌ Transcript with ID '{id_reference}' not found.")
            return
        
        clean_id = file_manager.id_generator.parse_reference_id(id_reference)
        audio_files_info = file_manager.get_audio_files_for_transcript(id_reference)
        
        print(f"\n🎵 Audio files for transcript {clean_id}:")
        
        if not audio_files_info:
            print("No audio files found for this transcript.")
            return
        
        print("-" * 80)
        print(f"{'Filename':<40} {'Size':<10} {'Duration':<10} {'Status'}")
        print("-" * 80)
        
        total_size = 0
        total_duration = 0
        
        for audio_info in audio_files_info:
            filename = os.path.basename(audio_info['path'])
            size_str = f"{audio_info['size_mb']:.1f}MB" if audio_info['exists'] else "Missing"
            duration_str = f"{audio_info['duration']:.1f}s" if audio_info['exists'] else "-"
            status = "✅ OK" if audio_info['exists'] else "❌ Missing"
            
            print(f"{filename:<40} {size_str:<10} {duration_str:<10} {status}")
            
            if audio_info['exists']:
                total_size += audio_info['size_mb']
                total_duration += audio_info['duration']
        
        print("-" * 80)
        print(f"Total: {len(audio_files_info)} files, {total_size:.1f}MB, {total_duration:.1f}s")
        
    except Exception as e:
        print(f"❌ Error showing audio files: {e}")

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
        new_transcript, session_audio_file = record_audio_streaming()
        if not new_transcript:
            print("❌ No new content recorded.")
            return
        
        # Deduplicate the new content
        new_transcript = deduplicate_transcript(new_transcript)
        
        print("\n--- NEW CONTENT ---")
        print(new_transcript)
        print("--------------------")
        
        # Append to existing transcript
        updated_path = file_manager.append_to_transcript(id_reference, new_transcript, session_audio_file=session_audio_file)
        
        if updated_path:
            print(f"✅ Successfully appended to transcript {clean_id}")
            print(f"📁 Updated file: {updated_path}")
            
            # Clean up the session file after successful append
            if session_audio_file and session_audio_file.exists():
                try:
                    session_audio_file.unlink()  # Remove the temporary session file
                except Exception as e:
                    print(f"⚠️ Could not remove session file: {e}")
            
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
            ollama_api_url=OLLAMA_API_URL,
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
                    ollama_api_url=OLLAMA_API_URL,
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
        
        # Use streaming transcription (now the default and only mode)
        verbose = (hasattr(args, 'verbose') and args.verbose) or STREAMING_VERBOSE
        
        print("🚀 Starting streaming transcription")
        
        # 1. Record Audio with streaming system
        device_override = args.device if hasattr(args, 'device') and args.device is not None else None
        transcription_result = record_audio_streaming(device_override, verbose)
        
        if len(transcription_result) == 4:
            transcribed_text, master_audio_file, existing_file_path, existing_transcript_id = transcription_result
        else:
            # Fallback for compatibility
            transcribed_text, master_audio_file = transcription_result
            existing_file_path, existing_transcript_id = None, None
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
        print("📋 Transcription copied to clipboard.")

    # 4. Save transcript with default filename first, then let AI rename it
    # Check if streaming already created a transcript file
    if existing_file_path and existing_transcript_id:
        file_path = existing_file_path  
        transcript_id = existing_transcript_id
    else:
        # This is the fallback transcription path - create new transcript
        print("�💾 Saving fallback transcript and audio...")
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        
        try:
            file_path, transcript_id = file_manager.create_new_transcript(
                transcribed_text, 
                "transcript",  # Use default name initially
                session_audio_file=master_audio_file
            )
            print(f"✅ Transcript {transcript_id} saved")
            
            # Clean up the session file after successful storage
            if master_audio_file and master_audio_file.exists():
                try:
                    master_audio_file.unlink()  # Remove the temporary session file
                except Exception as e:
                    print(f"⚠️ Could not remove session file: {e}")
                    
        except Exception as e:
            print(f"❌ Error saving transcript: {e}")
            return

    # 5. Add AI-generated summary, tags, and proper filename (if enabled)
    if AUTO_METADATA and transcribed_text and transcribed_text.strip():
        print("🤖 Generating summary and tags...")
        summarizer = SummarizationService(
            ollama_model=OLLAMA_MODEL,
            ollama_api_url=OLLAMA_API_URL,
            ollama_timeout=OLLAMA_TIMEOUT,
            max_content_length=OLLAMA_MAX_CONTENT_LENGTH
        )
        if summarizer.check_ollama_available():
            success = summarizer.summarize_file(file_path, copy_to_notes=False)
            if success:
                print("✅ Summary and tags added to transcript metadata")
                
                # Background enhancement (full audio transcription + cleanup) starts now
                print("🔄 Starting background: full transcription → enhanced summary → audio cleanup...")
            else:
                print("⚠️ Could not generate AI summary - transcript saved without metadata")
        else:
            print("ℹ️  Ollama not available - transcript saved without AI metadata")

    # 6. Handle post-transcription actions
    summarizer_for_check = SummarizationService(
        ollama_model=OLLAMA_MODEL,
        ollama_api_url=OLLAMA_API_URL,
        ollama_timeout=OLLAMA_TIMEOUT,
        max_content_length=OLLAMA_MAX_CONTENT_LENGTH
    )
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
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose monitoring output for streaming transcription')
    parser.add_argument('id_reference', nargs='?', 
                       help='Reference existing transcript by ID (e.g., -123456)')
    parser.add_argument('-l', '--list', action='store_true',
                       help='List all transcripts with their IDs')
    parser.add_argument('-v', '--view', type=str, metavar='ID', dest='show',
                       help='Show content of transcript by ID')
    parser.add_argument('-g', '--genai', type=str, metavar='PATH_OR_ID', dest='summarize',
                       help='AI analysis and tagging of a file by path or transcript ID (e.g., /path/to/file.md or -123)')
    parser.add_argument('--audio', type=str, metavar='ID', dest='show_audio',
                       help='Show audio files associated with transcript by ID')
    parser.add_argument('--reprocess', type=str, metavar='ID', dest='reprocess',
                       help='Reprocess all audio files for transcript ID (transcribe + summarize)')
    parser.add_argument('--reprocess-failed', action='store_true',
                       help='Reprocess all orphaned audio files (audio without transcript)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing transcript when reprocessing (default: create new)')
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
        elif args.show_audio:
            show_audio_files(args.show_audio)
        elif args.reprocess:
            reprocess_transcript_command(args.reprocess, args.overwrite)
        elif args.reprocess_failed:
            reprocess_failed_command()
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