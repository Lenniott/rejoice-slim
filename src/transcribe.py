# src/transcribe.py

# Suppress urllib3 SSL warnings before any imports
import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

import os
import sys
import subprocess
import sounddevice as sd
import numpy as np
import time
from dotenv import load_dotenv
import tempfile
import wave
from pathlib import Path
import pyperclip
import argparse
import threading
import logging
import signal
import select
from typing import Optional, Dict, Any, Tuple

# Import our new ID-based transcript management
from transcript_manager import TranscriptFileManager, AudioFileManager
from id_generator import TranscriptIDGenerator
from file_header import TranscriptHeader

# Import summarization service
from summarization_service import get_summarizer

# Import streaming transcription components
from audio_buffer import CircularAudioBuffer
from volume_segmenter import VolumeSegmenter, SegmentProcessor, VolumeConfig
from quick_transcript import QuickTranscriptAssembler
from loading_indicator import LoadingIndicator
from safety_net import SafetyNetManager
from debug_logger import DebugLogger
from recording_health_monitor import RecordingHealthMonitor
from ui_display import (
    clear_screen, show_loading, show_recording,
    show_transcribing, show_processing, show_complete
)

# Import whisper engine (supports both faster-whisper and openai-whisper)
import whisper_engine as whisper

# Import settings module
from settings import settings_menu

# Import Obsidian utilities
from obsidian_utils import open_in_obsidian, is_path_in_vault

# Import commands module
from commands import (
    open_transcripts_folder,
    list_transcripts,
    show_audio_files,
    show_transcript,
    append_to_transcript,
    summarize_file,
    reprocess_transcript_command,
    reprocess_failed_command,
    list_recovery_sessions,
    recover_session
)

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
OBSIDIAN_ENABLED = os.getenv("OBSIDIAN_ENABLED", "false").lower() == "true"
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "")
AUTO_METADATA = os.getenv("AUTO_METADATA", "false").lower() == "true"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))  # Default 3 minutes for local LLMs
SAMPLE_RATE = 16000 # 16kHz is standard for Whisper

# Streaming transcription configuration (now the default)
STREAMING_BUFFER_SIZE_SECONDS = int(os.getenv("STREAMING_BUFFER_SIZE_SECONDS", "300"))  # 5 minutes
STREAMING_MIN_SEGMENT_DURATION = int(os.getenv("STREAMING_MIN_SEGMENT_DURATION", "30"))  # 30 seconds
STREAMING_TARGET_SEGMENT_DURATION = int(os.getenv("STREAMING_TARGET_SEGMENT_DURATION", "60"))  # 60 seconds
STREAMING_MAX_SEGMENT_DURATION = int(os.getenv("STREAMING_MAX_SEGMENT_DURATION", "90"))  # 90 seconds

# Audio device configuration
DEFAULT_MIC_DEVICE = int(os.getenv("DEFAULT_MIC_DEVICE", "-1"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress verbose logging from our libraries
# (Old audio_chunker, transcription_worker, vad_service logging removed - modules deleted)

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
# Old AI system removed - now using hierarchical SummarizationService only
# Settings functions moved to settings.py module
# Global cancellation state - accessible from signal handler
_global_recording_state = {
    'recording_event': None,
    'cancelled': None,
    'original_handler': None
}

# Session management helper functions
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

def safe_cleanup_audio(master_audio_file: Path, transcription_text: str,
                       recording_duration: float, sample_rate: int = 16000) -> bool:
    """
    Safely validate before deleting audio files to prevent data loss.

    Args:
        master_audio_file: Path to the master audio file
        transcription_text: The transcribed text
        recording_duration: Duration of recording in seconds
        sample_rate: Audio sample rate (default 16000)

    Returns:
        bool: True if safe to delete, False if files should be preserved
    """
    # Validation 1: Transcription content
    if not transcription_text or len(transcription_text.strip()) < 50:
        print("⚠️  Keeping audio - transcription appears empty")
        return False

    # Validation 2: File size check
    if master_audio_file and master_audio_file.exists():
        file_size = master_audio_file.stat().st_size
        expected_size = recording_duration * sample_rate * 2  # 2 bytes per sample (16-bit)
        if file_size < expected_size * 0.8:  # Less than 80% of expected size
            print(f"⚠️  Keeping audio - file size suspicious ({file_size/1024/1024:.1f} MB vs {expected_size/1024/1024:.1f} MB expected)")
            return False

        # Validation 3: WAV integrity
        try:
            with wave.open(str(master_audio_file), 'rb') as wav:
                frames = wav.getnframes()
                if frames < sample_rate * 10:  # Less than 10 seconds
                    print("⚠️  Keeping audio - WAV file too short")
                    return False
        except Exception as e:
            print(f"⚠️  Keeping audio - WAV file invalid: {e}")
            return False

    # Validation 4: Transcription quality (words per minute check)
    if recording_duration > 0:
        words = len(transcription_text.split())
        wpm = words / (recording_duration / 60)
        if wpm < 10:  # Less than 10 words per minute is suspicious
            print(f"⚠️  Keeping audio - suspicious transcription rate ({wpm:.1f} wpm)")
            return False

    return True


def record_audio_streaming(device_override: Optional[int] = None, debug: bool = False) -> Tuple[Optional[str], Optional[Path], Optional[str], Optional[str]]:
    """
    Records audio using the new streaming transcription system.
    Simplified version using sounddevice for compatibility.

    Returns:
        Tuple[Optional[str], Optional[Path], Optional[str], Optional[str]]: (transcribed_text, master_audio_file_path, transcript_path, transcript_id)
    """
    # Suppress all logging unless debug mode is enabled
    if not debug:
        import logging
        logging.getLogger('audio_buffer').setLevel(logging.ERROR)
        logging.getLogger('volume_segmenter').setLevel(logging.ERROR)
        logging.getLogger('safety_net').setLevel(logging.ERROR)
        logging.getLogger('quick_transcript').setLevel(logging.ERROR)
        logging.getLogger('audio_manager').setLevel(logging.ERROR)
        import warnings
        warnings.filterwarnings("ignore")
    
    # Show initialization indicator
    if not debug:
        indicator = LoadingIndicator("Initializing recording system...")
        indicator.start()
    
    # Generate session ID
    session_id = f"stream_{int(time.time())}"
    
    # Initialize debug logger
    debug_log = DebugLogger(session_id, SAVE_PATH or ".", enabled=debug)
    debug_log.milestone("Recording session started")
    
    # Create master audio file for safety net
    temp_audio_dir = Path(SAVE_PATH or tempfile.gettempdir()) / "audio_sessions"
    temp_audio_dir.mkdir(exist_ok=True)
    master_audio_file = temp_audio_dir / f"{session_id}.wav"
    
    # Determine audio device first (before try block so it's accessible everywhere)
    device = device_override if device_override is not None else (None if DEFAULT_MIC_DEVICE == -1 else DEFAULT_MIC_DEVICE)
    
    # Detect native sample rate of the device (critical for hardware compatibility)
    try:
        device_info = sd.query_devices(device, 'input')
        native_sample_rate = int(device_info['default_samplerate'])
        debug_log.detail(f"Detected native sample rate: {native_sample_rate} Hz for device: {device_info['name']}")
    except Exception as e:
        # Fallback to common Mac default
        native_sample_rate = 48000
        debug_log.detail(f"Could not detect sample rate, using default: {native_sample_rate} Hz")
    
    # Calculate resampling ratio if needed
    resample_ratio = SAMPLE_RATE / native_sample_rate if native_sample_rate != SAMPLE_RATE else 1.0
    needs_resampling = resample_ratio != 1.0
    
    if needs_resampling:
        debug_log.detail(f"Will resample from {native_sample_rate} Hz to {SAMPLE_RATE} Hz (ratio: {resample_ratio:.4f})")
    
    # Initialize components
    try:
        
        # Audio buffer (5-minute rolling buffer) - always use Whisper's expected rate
        audio_buffer = CircularAudioBuffer(
            capacity_seconds=STREAMING_BUFFER_SIZE_SECONDS,
            sample_rate=SAMPLE_RATE,
            channels=1
        )
        
        # Volume segmenter with configurable durations
        volume_config = VolumeConfig(
            min_segment_duration=STREAMING_MIN_SEGMENT_DURATION,
            target_segment_duration=STREAMING_TARGET_SEGMENT_DURATION,
            max_segment_duration=STREAMING_MAX_SEGMENT_DURATION
        )
        volume_segmenter = VolumeSegmenter(
            audio_buffer=audio_buffer,
            config=volume_config,
            verbose=debug
        )
        
        debug_log.detail(f"Initialized volume segmenter: min={STREAMING_MIN_SEGMENT_DURATION}s, target={STREAMING_TARGET_SEGMENT_DURATION}s, max={STREAMING_MAX_SEGMENT_DURATION}s")
        
        # Initialize audio segment processor
        segment_extractor = SegmentProcessor(audio_buffer)
        
        # Initialize safety net
        safety_net = SafetyNetManager(
            safety_log_path=os.path.join(SAVE_PATH or ".", "safety_log.json"),
            auto_recovery=True,
            max_retry_attempts=3
        )
        
        # Start safety net session
        safety_record = safety_net.start_session(session_id, str(master_audio_file))
        
        # Quick transcript assembler
        file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)
        audio_manager = AudioFileManager(SAVE_PATH)
        assembler = QuickTranscriptAssembler(
            transcript_manager=file_manager,
            audio_manager=audio_manager,
            auto_clipboard=AUTO_COPY
        )
        
        # Initialize components without logging details
        debug_log.detail("All streaming components initialized successfully")
        
    except Exception as e:
        if not debug:
            indicator.stop()
        debug_log.error(f"Failed to initialize streaming components: {e}")
        print(f"❌ Failed to initialize streaming components: {e}")
        return None, None, None, None
    
    # Audio file writer for master file
    audio_writer = None
    recording_active = threading.Event()
    recording_active.set()
    user_stopped = threading.Event()

    # Thread-safe volume level tracking for UI meter
    current_volume_rms = [0.0]  # Use list for thread-safe sharing
    volume_lock = threading.Lock()

    # Initialize health monitor for recording
    health_monitor = RecordingHealthMonitor()
    debug_log.detail("Health monitor initialized")
    
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
        if not debug:
            indicator.stop()
        debug_log.error("Failed to initialize audio writer")
        return None, None, None, None
    
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
        """Process incoming audio data with health monitoring."""
        if not recording_active.is_set():
            return

        bytes_written = 0
        write_success = False

        try:
            # Convert to flat mono array
            audio_data = indata.flatten()

            # Calculate RMS volume for UI meter (0.0 to 1.0 range)
            rms = np.sqrt(np.mean(audio_data ** 2))
            # Normalize to reasonable range (typical speech is 0.01-0.3)
            # Increased sensitivity: 5.0x multiplier for more responsive meter
            normalized_volume = min(rms * 5.0, 1.0)

            # Update shared volume level (thread-safe)
            with volume_lock:
                current_volume_rms[0] = normalized_volume

            # Resample if needed (from native rate to Whisper's 16kHz)
            if needs_resampling:
                # Simple linear interpolation resampling
                original_length = len(audio_data)
                target_length = int(original_length * resample_ratio)
                resampled_audio = np.interp(
                    np.linspace(0, original_length, target_length),
                    np.arange(original_length),
                    audio_data
                )
                audio_data_16k = resampled_audio
            else:
                audio_data_16k = audio_data

            # Write to master file immediately (at 16kHz for Whisper compatibility)
            audio_16bit = (audio_data_16k * 32767).astype(np.int16)
            audio_bytes = audio_16bit.tobytes()

            # Track file position before write
            bytes_before = audio_writer._file.tell() if audio_writer else 0

            audio_writer.writeframes(audio_bytes)
            audio_writer._file.flush()

            # Track bytes written
            bytes_after = audio_writer._file.tell() if audio_writer else 0
            bytes_written = bytes_after - bytes_before
            write_success = bytes_written > 0

            # Add to circular buffer for streaming (already resampled to 16kHz)
            audio_buffer.write(audio_data_16k)

            # Optional debug monitoring
            if debug and hasattr(audio_callback, 'call_count'):
                audio_callback.call_count += 1
                if audio_callback.call_count % 100 == 0:  # Every ~2 seconds
                    duration = audio_callback.call_count * frames / native_sample_rate
                    print(f"🎙️  Recording: {duration:.1f}s", end='\r', flush=True)
            elif debug:
                audio_callback.call_count = 1

        except Exception as e:
            debug_log.error(f"Audio callback error: {e}")
            if debug:
                print(f"⚠️ Audio callback error: {e}")
            write_success = False

        finally:
            # Always track callback health (even on failure)
            health_monitor.track_callback(len(indata), bytes_written, write_success)
    
    try:
        # Start keyboard listener
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        # Stop initialization indicator
        if not debug:
            indicator.stop()
        
        debug_log.milestone("Recording initialized, starting audio capture")

        # Show initial recording screen (unless in debug mode)
        if not debug:
            show_recording(0, 0.0)
        else:
            # Platform-specific instructions for debug mode
            print("🔴 Recording started")
            if sys.platform == "darwin":  # macOS
                print("   Press Enter to stop, or Ctrl+C (^C) to cancel")
            else:
                print("   Press Enter to stop, or Ctrl+C to cancel")
            print("   Status updates every 5 seconds...\n")
        
        
        
        # Start recording with sounddevice
        audio_buffer.start_recording()
        volume_segmenter.start_analysis()
        assembler.start_session(session_id)
        
        # Open stream at native sample rate (what hardware supports)
        # Try the specified device, with fallback to other devices if it fails
        stream = None
        tried_devices = []
        
        # Get list of available input devices for fallback
        available_devices = []
        all_devices = sd.query_devices()
        for i, dev in enumerate(all_devices):
            if dev['max_input_channels'] > 0:
                available_devices.append(i)
        
        # Build list of devices to try (primary device first, then fallbacks)
        devices_to_try = []
        if device is not None:
            devices_to_try.append(device)
        # Add other available devices as fallbacks
        for fallback_dev in available_devices:
            if fallback_dev not in devices_to_try:
                devices_to_try.append(fallback_dev)
        
        # Try each device until one works
        for try_device in devices_to_try:
            try:
                tried_devices.append(try_device)
                device_name = sd.query_devices(try_device)['name']
                
                if debug:
                    print(f"🎤 Trying device {try_device}: {device_name}")
                
                stream = sd.InputStream(
                    samplerate=native_sample_rate, 
                    channels=1, 
                    callback=audio_callback,
                    device=try_device,
                    blocksize=1024
                )
                stream.start()
                
                # Success!
                debug_log.detail(f"Successfully opened audio stream on device {try_device}: {device_name}")
                if try_device != device and device is not None:
                    print(f"ℹ️  Using {device_name} (device {try_device}) - your default device was unavailable")
                break
                
            except Exception as e:
                if stream:
                    try:
                        stream.close()
                    except:
                        pass
                    stream = None
                
                if debug or try_device == devices_to_try[-1]:  # Show error on last attempt
                    debug_log.detail(f"Failed to open device {try_device}: {e}")
                
                # If this was the last device, raise the error
                if try_device == devices_to_try[-1]:
                    raise Exception(
                        f"Could not open any audio input device. Tried devices: {tried_devices}. "
                        f"Last error: {e}. "
                        "Try: 1) Close apps using your microphone (Zoom, Teams, etc.), "
                        "2) Check System Settings → Privacy & Security → Microphone, "
                        "3) Use 'rec -s' to select a different microphone device."
                    )
        
        if not stream:
            raise Exception("Failed to open audio stream after trying all available devices")
        
        # Monitor recording with live feedback
        start_time = time.time()
        last_status_time = time.time()
        last_health_check = time.time()
        segments_processed = 0

        while recording_active.is_set():
            current_time = time.time()
            duration = current_time - start_time

            # NEW: Periodic health check every 10 seconds
            if current_time - last_health_check > 10.0:
                health_status = health_monitor.check_health()

                if health_status['is_critical']:
                    # CRITICAL: Stop recording immediately
                    print(f"\n❌ RECORDING FAILURE DETECTED")
                    for issue in health_status['issues']:
                        print(f"   • {issue}")
                    print("⚠️  Stopping to prevent data loss...")
                    debug_log.error(f"Critical recording failure: {health_status['issues']}")
                    recording_active.clear()
                    break
                elif health_status['issues']:
                    # WARNING: Show but continue
                    print(f"\n⚠️  Recording warning:")
                    for issue in health_status['issues']:
                        print(f"   • {issue}")
                    debug_log.warning(f"Recording warnings: {health_status['issues']}")

                # Check master file growth
                if master_audio_file.exists():
                    current_size = master_audio_file.stat().st_size
                    expected_size = duration * SAMPLE_RATE * 2  # 2 bytes per sample

                    if current_size < expected_size * 0.5 and duration > 30:
                        print(f"\n⚠️  Master file not growing properly")
                        print(f"   Current: {current_size/1024/1024:.1f} MB")
                        print(f"   Expected: {expected_size/1024/1024:.1f} MB")
                        debug_log.warning(f"Master file growth issue: {current_size} vs {expected_size} expected")

                last_health_check = current_time

            # NEW: Status update every 
            if current_time - last_status_time > 0.5:
                stats = health_monitor.get_stats()
                if not debug:
                    # Clean UI: show recording screen with updated time
                    # Get real-time volume from audio callback
                    with volume_lock:
                        volume_level = current_volume_rms[0]
                    show_recording(duration, volume_level)
                else:
                    # Debug mode: show detailed status line
                    print(f"\r🔴 Recording: {int(duration//60):02d}:{int(duration%60):02d} | "
                          f"📊 {stats['total_mb']:.1f} MB | "
                          f"✓ {segments_processed} segments",
                          end='', flush=True)
                last_status_time = current_time

            # Process completed segments
            try:
                # First, analyze audio to detect new segments
                new_segments = volume_segmenter.analyze_and_segment()
                debug_log.detail(f"Checked for new segments, found {len(new_segments)}")
                if debug and new_segments:
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
                            debug_log.detail(f"Processed segment {segments_processed}: {segment_info.duration:.1f}s")
                            if debug:
                                print(f"\n📦 Processed segment {segments_processed}: {segment_info.duration:.1f}s")
                        else:
                            debug_log.warning(f"Could not extract audio for segment {segments_processed}")
                            if debug:
                                print(f"⚠️ Could not extract audio for segment {segments_processed}")
                    except Exception as e:
                        debug_log.error(f"Segment processing error: {e}")
                        if debug:
                            print(f"⚠️ Segment processing error: {e}")
            except Exception as e:
                debug_log.error(f"Segmentation error: {e}")
                if debug:
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

        # Get actual recording duration
        recording_duration = audio_buffer.get_recording_duration()
        debug_log.milestone(f"Recording stopped: {recording_duration:.1f}s captured")
        debug_log.detail(f"Total audio duration: {recording_duration:.1f}s")

        # NEW: Validate recording before proceeding
        print("\n\n🔍 Validating recording...")
        validation_issues = []

        # Check master file exists and has expected size
        if not master_audio_file.exists():
            validation_issues.append("Master audio file missing!")
        else:
            file_size = master_audio_file.stat().st_size
            expected_size = recording_duration * SAMPLE_RATE * 2  # 2 bytes per sample
            size_ratio = file_size / expected_size if expected_size > 0 else 0

            if size_ratio < 0.5:
                validation_issues.append(f"Master file too small ({size_ratio*100:.1f}% of expected)")
            elif size_ratio < 0.8:
                validation_issues.append(f"Master file smaller than expected ({size_ratio*100:.1f}%)")

            # Validate WAV integrity
            try:
                with wave.open(str(master_audio_file), 'rb') as wav:
                    frames = wav.getnframes()
                    if frames < SAMPLE_RATE * 5:
                        validation_issues.append("WAV file too short (less than 5 seconds)")
            except Exception as e:
                validation_issues.append(f"WAV file corrupted: {e}")

        if validation_issues:
            print("❌ VALIDATION ISSUES DETECTED:")
            for issue in validation_issues:
                print(f"   • {issue}")
            print("\n⚠️  Audio files will be PRESERVED for recovery")
            print(f"📁 Master audio: {master_audio_file}")
            debug_log.error(f"Validation failed: {validation_issues}")

            try:
                user_response = input("\nAttempt transcription anyway? (y/n): ").strip().lower()
                if user_response != 'y':
                    print("Exiting without transcription. Audio preserved for later recovery.")
                    return None, master_audio_file, None, None
            except (EOFError, KeyboardInterrupt):
                print("\nExiting without transcription. Audio preserved.")
                return None, master_audio_file, None, None
        else:
            print("✓ Recording validated successfully")

        # Start loading indicator for processing
        loader = LoadingIndicator("🎤 Processing audio...")
        loader.start()
        
        # Decide on processing strategy based on duration
        use_streaming = recording_duration >= 90.0
        debug_log.detail(f"Processing strategy: {'streaming + full' if use_streaming else 'full only'} (duration: {recording_duration:.1f}s)")
        
        try:
            quick_transcript_text = None
            quick_transcript_path = None
            quick_transcript_id = None
            streaming_attempt = None  # Will be set if we use streaming
            
            if use_streaming:
                # Register streaming attempt only for long recordings
                streaming_attempt = safety_net.register_processing_attempt(
                    session_id, "streaming", {"buffer_size": STREAMING_BUFFER_SIZE_SECONDS}
                )
                
                # For recordings >= 90s: Process streaming segments first for quick transcript
                loader.update("🔍 Analyzing segments...")
                debug_log.milestone("Processing quick transcript from streaming segments")
                
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
                                debug_log.detail(f"Final segment: {segment_info.duration:.1f}s")
                                if debug:
                                    print(f"📦 Final segment: {segment_info.duration:.1f}s")
                            else:
                                debug_log.warning("Could not extract final segment audio")
                                if debug:
                                    print(f"⚠️ Could not extract final segment audio")
                        except Exception as e:
                            debug_log.error(f"Final segment error: {e}")
                            if debug:
                                print(f"⚠️ Final segment error: {e}")
                except Exception as e:
                    debug_log.error(f"Final processing error: {e}")
                    if debug:
                        print(f"⚠️ Final processing error: {e}")
                
                # Finalize quick transcript
                debug_log.detail("📝 Finalizing quick transcript...")
                quick_transcript = assembler.finalize_transcript()

                # Check if we have real transcription content
                if quick_transcript and quick_transcript.has_content:
                    quick_transcript_text = quick_transcript.transcript_text.strip()
                    quick_transcript_path = quick_transcript.file_path
                    quick_transcript_id = quick_transcript.transcript_id
                    debug_log.milestone(f"Quick transcript ready: {len(quick_transcript_text)} chars")

                    # Copy to clipboard immediately
                    if AUTO_COPY and quick_transcript_text:
                        loader.update("📋 Copying to clipboard...")
                        import pyperclip
                        pyperclip.copy(quick_transcript_text)
                        loader.stop("📋 Quick transcript copied to clipboard.")
                        debug_log.detail("Quick transcript copied to clipboard")
                    else:
                        loader.stop()

                    safety_net.complete_processing_attempt(
                        session_id, streaming_attempt,
                        output_files=[quick_transcript_path] if quick_transcript_path else [],
                        success=True
                    )
                else:
                    debug_log.warning("Quick transcript had no content, skipping to full transcription")
                    safety_net.complete_processing_attempt(
                        session_id, streaming_attempt, success=False,
                        error_message="No quick content"
                    )
            else:
                # For <90s recordings: Skip streaming entirely
                debug_log.milestone("Skipping streaming for short recording, going directly to full transcription")
                # No streaming attempt registered, so nothing to complete
            
            # Now run full-file Whisper transcription (for all recordings)
            if not debug:
                show_transcribing(0, session_id)
            debug_log.milestone("🔄 Running full audio transcription...")
            debug_log.milestone("Starting full Whisper transcription")

            transcription_start_time = time.time()

            # Start progress updater for transcription (conservative estimate based on audio duration)
            stop_transcribe_progress = threading.Event()

            # Estimate transcription time based on model size and audio duration
            # faster-whisper performance varies by model:
            # - tiny/base: ~0.1-0.15x audio duration
            # - small: ~0.2-0.3x audio duration
            # - medium: ~0.4-0.6x audio duration
            # - large: ~0.8-1.2x audio duration
            model_multipliers = {
                'tiny': 0.15,
                'base': 0.2,
                'small': 0.3,
                'medium': 0.5,
                'large': 1.0,
                'large-v2': 1.0,
                'large-v3': 1.0
            }
            multiplier = model_multipliers.get(WHISPER_MODEL.lower(), 0.3)
            estimated_transcribe_time = max(recording_duration * multiplier, 5.0) if recording_duration > 0 else 20

            def transcribe_progress_updater():
                while not stop_transcribe_progress.is_set():
                    if not debug:
                        elapsed = time.time() - transcription_start_time
                        # Conservative progress: caps at 95% until complete
                        progress = min(int((elapsed / estimated_transcribe_time) * 95), 95)
                        show_transcribing(elapsed, session_id, progress=progress)
                    time.sleep(0.5)  # Update every 0.5 seconds

            transcribe_thread = threading.Thread(target=transcribe_progress_updater, daemon=True)
            transcribe_thread.start()

            try:
                whisper_model = whisper.load_model(WHISPER_MODEL)
                debug_log.detail(f"Loaded Whisper model: {WHISPER_MODEL}")

                if WHISPER_LANGUAGE and WHISPER_LANGUAGE.lower() != "auto":
                    result = whisper_model.transcribe(str(master_audio_file), fp16=False, language=WHISPER_LANGUAGE)
                else:
                    result = whisper_model.transcribe(str(master_audio_file), fp16=False)

                full_transcript_text = result['text'].strip()

                # Stop progress updates
                stop_transcribe_progress.set()
                transcribe_thread.join(timeout=1)

                transcription_elapsed = time.time() - transcription_start_time
                debug_log.milestone(f"Full transcription complete: {len(full_transcript_text)} chars")

                # Always show 100% when complete
                if not debug:
                    show_transcribing(transcription_elapsed, session_id, progress=100)

                if full_transcript_text:
                    # Mark full transcription as successful
                    full_attempt = safety_net.register_processing_attempt(
                        session_id, "full", {"audio_file": str(master_audio_file)}
                    )
                    safety_net.complete_processing_attempt(
                        session_id, full_attempt, success=True,
                        output_files=[str(master_audio_file)]
                    )
                    safety_net.complete_session(session_id, success=True,
                                               final_output_files=[str(master_audio_file)])

                    loader.stop()
                    debug_log.milestone("Transcription complete, returning results")
                    
                    # Return full transcript (it will be saved and processed by main())
                    return full_transcript_text, master_audio_file, quick_transcript_path, quick_transcript_id
                else:
                    debug_log.error("Full transcription returned empty text")
                    safety_net.complete_session(session_id, success=False)
                    loader.stop()
                    return None, None, None, None
                    
            except Exception as e:
                debug_log.error(f"Full transcription failed: {e}")
                safety_net.complete_session(session_id, success=False)
                loader.stop()
                return None, None, None, None
        
        except Exception as e:
            loader.stop()  # Clear loading indicator on error
            raise
            
    except KeyboardInterrupt:
        return None, None, None, None
        
    except Exception as e:
        # Log the actual error
        print(f"\n❌ Recording failed: {e}")
        if 'debug_log' in locals():
            debug_log.error(f"Exception during recording: {e}")
        
        # Only complete streaming attempt if it was registered
        if 'streaming_attempt' in locals() and streaming_attempt is not None:
            safety_net.complete_processing_attempt(
                session_id, streaming_attempt, success=False,
                error_message=str(e)
            )
        safety_net.complete_session(session_id, success=False)
        return None, None, None, None
    
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

    def open_file(file_path):
        """Open file with Obsidian if enabled, otherwise use default app"""
        # Try to open in Obsidian using URI scheme if enabled
        if file_path.endswith('.md') and OBSIDIAN_ENABLED and OBSIDIAN_VAULT_PATH:
            from obsidian_utils import open_in_obsidian, is_path_in_vault

            # Check if file is within configured vault
            if is_path_in_vault(file_path, OBSIDIAN_VAULT_PATH):
                # Try opening with Obsidian URI
                if open_in_obsidian(file_path, OBSIDIAN_VAULT_PATH):
                    return  # Successfully opened in Obsidian

        # Fall back to default app
        if sys.platform == "darwin":
            subprocess.run(["open", file_path])
        else:
            subprocess.run(["xdg-open", file_path])

    # Open file - only if explicitly enabled
    if should_open:
        open_file(full_path)
    elif not hasattr(args, 'open') or args.open is None:
        # Only ask if AUTO_OPEN is not explicitly set to false
        if not AUTO_OPEN:
            # Don't ask, just skip
            pass
        else:
            # Ask user
            if input("📂 Open the file? (y/n): ").lower() == 'y':
                open_file(full_path)
                
def main(args=None):
    # Track if debug mode for UI decisions
    debug = False
    session_id = None
    recording_duration = 0

    try:
        # Set defaults if no args provided
        if args is None:
            args = type('Args', (), {})()

        # Use streaming transcription (now the default and only mode)
        # Support both --debug and --verbose (deprecated) flags
        debug = (hasattr(args, 'debug') and args.debug) or (hasattr(args, 'verbose') and args.verbose)

        # Show loading screen (unless in debug mode)
        if not debug:
            show_loading()

        # 1. Record Audio with streaming system
        device_override = args.device if hasattr(args, 'device') and args.device is not None else None
        transcription_result = record_audio_streaming(device_override, debug)
        
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

    # 3. Copy full transcript to clipboard (if not already copied by quick transcript)
    if AUTO_COPY and not existing_file_path:  # Only copy if quick didn't already
        pyperclip.copy(transcribed_text)
        if debug:
            print("📋 Transcription copied to clipboard.\n\n")

    # 4. Save/update transcript with full transcription
    file_manager = TranscriptFileManager(SAVE_PATH, OUTPUT_FORMAT)

    if existing_file_path and existing_transcript_id:
        file_path = existing_file_path
        transcript_id = existing_transcript_id
        if debug:
            print(f"💾 Updating transcript {transcript_id} with full transcription...")

        try:
            # Update the transcript content
            success = file_manager.update_transcript_content(file_path, transcribed_text)
            if success and debug:
                print(f"✅ Full transcription saved to {transcript_id}")
            elif not success:
                if debug:
                    print(f"⚠️ Could not update transcript, creating new one...")
                existing_file_path = None
        except Exception as e:
            if debug:
                print(f"⚠️ Error updating transcript: {e}, creating new one...")
            existing_file_path = None

    stored_audio_path = None
    if not existing_file_path:
        # Create new transcript
        if debug:
            print("💾 Saving transcript...")

        try:
            file_path, transcript_id, stored_audio_path = file_manager.create_new_transcript(
                transcribed_text,
                "transcript",  # Use default name initially
                session_audio_file=master_audio_file
            )
            if debug:
                print(f"✅ Transcript {transcript_id} saved")
        except Exception as e:
            print(f"❌ Error saving transcript: {e}")
            return
    
    # Mark transcript as processed (full transcription complete)
    try:
        file_manager.update_transcript_status(file_path, "processed")
    except Exception as e:
        print(f"⚠️ Could not update transcript status: {e}")

    # 5. Add AI-generated summary, tags, and proper filename (if enabled)
    final_file_path = file_path  # Track the final path (may change if renamed)
    processing_start_time = time.time()

    if AUTO_METADATA and transcribed_text and transcribed_text.strip():
        if not debug:
            show_processing(0, transcript_id, os.path.basename(file_path))

        if debug:
            print("🤖 Generating summary and tags...")

        summarizer = get_summarizer()
        if summarizer.check_ollama_available():
            # Estimate processing time based on audio duration (conservative: ~1.5x audio length)
            # This ensures progress moves slowly but always completes at 100%
            estimated_processing_time = recording_duration * 1.5 if recording_duration > 0 else 30

            # Update progress during processing
            def update_progress():
                if not debug:
                    elapsed = time.time() - processing_start_time
                    # Conservative progress: caps at 95% until complete
                    progress = min(int((elapsed / estimated_processing_time) * 95), 95)
                    show_processing(elapsed, transcript_id, os.path.basename(file_path), progress=progress)

            # Start background thread to update progress
            import threading
            stop_progress = threading.Event()
            def progress_updater():
                while not stop_progress.is_set():
                    update_progress()
                    time.sleep(0.5)  # Update every 0.5 seconds

            progress_thread = threading.Thread(target=progress_updater, daemon=True)
            progress_thread.start()

            # Run summarization
            success, new_path = summarizer.summarize_file(file_path, copy_to_notes=False)

            # Stop progress updates
            stop_progress.set()
            progress_thread.join(timeout=1)

            processing_elapsed = time.time() - processing_start_time

            # Always show 100% when complete
            if not debug:
                show_processing(processing_elapsed, transcript_id, os.path.basename(new_path) if new_path else os.path.basename(file_path), progress=100)

            if success:
                if debug:
                    print("✅ Summary and tags added to transcript metadata")
                # Update final path if file was renamed
                if new_path:
                    final_file_path = new_path
            else:
                if debug:
                    print("⚠️ Could not generate AI summary - transcript saved without metadata")
        else:
            if debug:
                print("ℹ️  Ollama not available - transcript saved without AI metadata")

    # Calculate recording duration for final screen
    if master_audio_file and master_audio_file.exists():
        try:
            with wave.open(str(master_audio_file), 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                recording_duration = frames / float(rate) if rate > 0 else 0
        except Exception:
            recording_duration = 0

    # Calculate word count
    word_count = len(transcribed_text.split())

    # Show complete screen (unless in debug mode)
    if not debug:
        show_complete(
            transcript_id,
            os.path.basename(final_file_path),
            final_file_path,
            recording_duration,
            word_count
        )

    # 6. Handle post-transcription actions (file opening)
    summarizer_for_check = get_summarizer()
    handle_post_transcription_actions(transcribed_text, final_file_path, summarizer_for_check.check_ollama_available(), args)

    # 7. Prompt user about audio cleanup at the very end
    # Calculate recording duration from audio file if it exists
    recording_duration = 0
    if master_audio_file and master_audio_file.exists():
        try:
            with wave.open(str(master_audio_file), 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                recording_duration = frames / float(rate) if rate > 0 else 0
        except Exception:
            # If we can't read the file, use a safe default that won't trigger cleanup
            recording_duration = 0

    # Validate transcription quality before offering cleanup
    cleanup_safe = safe_cleanup_audio(master_audio_file, transcribed_text, recording_duration, SAMPLE_RATE)

    if cleanup_safe:
        # Ask user if they want to remove audio files (default: yes)
        user_input = input("\n🗑️  Remove audio files? (Y/n): ").strip().lower()
        should_cleanup = user_input in ['', 'y', 'yes']

        if should_cleanup:
            files_cleaned = 0

            # Clean up session audio file (in audio_sessions/)
            if master_audio_file and master_audio_file.exists():
                try:
                    master_audio_file.unlink()
                    files_cleaned += 1
                except Exception as e:
                    print(f"⚠️ Could not remove session audio file: {e}")

            # Clean up stored audio file (in audio/)
            # Only needed for recovery if transcription fails - safe to delete after success
            if stored_audio_path and os.path.exists(stored_audio_path):
                try:
                    os.remove(stored_audio_path)
                    files_cleaned += 1
                except Exception as e:
                    print(f"⚠️ Could not remove stored audio file: {e}")

            if files_cleaned > 0:
                print(f"✅ Audio file{'s' if files_cleaned > 1 else ''} removed")
        else:
            print(f"💾 Audio files preserved:")
            if master_audio_file and master_audio_file.exists():
                print(f"   Session: {master_audio_file}")
            if stored_audio_path and os.path.exists(stored_audio_path):
                print(f"   Stored: {stored_audio_path}")
    else:
        # Validation failed - preserve audio files automatically
        print(f"\n💾 Audio files preserved for safety:")
        if master_audio_file and master_audio_file.exists():
            print(f"   Session: {master_audio_file}")
        if stored_audio_path and os.path.exists(stored_audio_path):
            print(f"   Stored: {stored_audio_path}")

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
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode with detailed logging to file and console milestones')
    parser.add_argument('--verbose', action='store_true',
                       help='(Deprecated: use --debug) Enable debug mode')
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
            list_transcripts(SAVE_PATH, OUTPUT_FORMAT)
        elif args.show:
            show_transcript(args.show, SAVE_PATH, OUTPUT_FORMAT)
        elif args.show_audio:
            show_audio_files(args.show_audio, SAVE_PATH, OUTPUT_FORMAT)
        elif args.reprocess:
            reprocess_transcript_command(args.reprocess, SAVE_PATH, OUTPUT_FORMAT, args.overwrite, AUTO_METADATA, WHISPER_MODEL)
        elif args.reprocess_failed:
            reprocess_failed_command(SAVE_PATH, OUTPUT_FORMAT, AUTO_METADATA, WHISPER_MODEL)
        elif args.summarize:
            summarize_file(args.summarize, SAVE_PATH, OUTPUT_FORMAT, AUTO_COPY)
        elif args.id_reference:
            append_to_transcript(args.id_reference, SAVE_PATH, OUTPUT_FORMAT, AUTO_COPY, record_audio_streaming, deduplicate_transcript)
        elif args.open_folder:
            open_transcripts_folder(SAVE_PATH)
        elif args.list_sessions:
            list_recovery_sessions(SAVE_PATH, SAMPLE_RATE)
        elif args.recover:
            recover_session(args.recover, SAVE_PATH, OUTPUT_FORMAT, SAMPLE_RATE, AUTO_METADATA, WHISPER_MODEL, transcribe_session_file)
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