# src/commands.py
"""
CLI command handlers for transcript management operations.
Extracted from transcribe.py for better modularity.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
import pyperclip

from transcript_manager import TranscriptFileManager
from file_header import TranscriptHeader
from summarization_service import SummarizationService, get_summarizer


def open_transcripts_folder(save_path):
    """Open the transcripts folder in Finder/Explorer."""
    try:
        if not save_path or not os.path.exists(save_path):
            print(f"❌ Transcripts folder not found: {save_path}")
            print("💡 Run 'rec -s' to configure the save path")
            return
        
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", save_path])
        elif sys.platform == "linux":  # Linux
            subprocess.run(["xdg-open", save_path])
        elif sys.platform == "win32":  # Windows
            subprocess.run(["explorer", save_path])
        else:
            print(f"📁 Transcripts folder: {save_path}")
            return
        
        print(f"📂 Opened transcripts folder: {save_path}")
        
    except Exception as e:
        print(f"❌ Error opening transcripts folder: {e}")
        print(f"📁 Transcripts folder location: {save_path}")


def list_transcripts(save_path, output_format):
    """List all available transcripts with their IDs."""
    try:
        file_manager = TranscriptFileManager(save_path, output_format)
        transcripts = file_manager.list_transcripts_with_audio()
        
        # Also check for legacy format files (only in SAVE_PATH)
        legacy_files = []
        if os.path.exists(save_path):
            for filename in os.listdir(save_path):
                if (filename.endswith(('.md', '.txt')) and 
                    TranscriptHeader.is_legacy_format_file(filename) and
                    not TranscriptHeader.is_id_format_file(filename)):
                    file_path = os.path.join(save_path, filename)
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
            print("📜 Legacy Format (timestamp-based):")
            for filename, mod_time in legacy_files:
                date_str = mod_time.strftime("%Y-%m-%d %H:%M")
                print(f"   {filename} ({date_str})")
            
        
        if transcripts:
            print(f"\n💡 Use 'rec -v ID' to view transcript content")
            print(f"💡 Use 'rec --audio ID' to see audio files for a transcript")
            print(f"💡 Use 'rec --reprocess ID' to reprocess all audio for a transcript")
        print(f"💡 New transcripts use format: ID_DDMMYYYY_descriptive-name.{output_format}")
        print(f"💡 Use 'rec --reprocess-failed' to process orphaned audio files")
        
    except Exception as e:
        print(f"❌ Error listing transcripts: {e}")


def show_audio_files(id_reference, save_path, output_format):
    """Show audio files associated with a transcript."""
    try:
        file_manager = TranscriptFileManager(save_path, output_format)
        
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


def show_transcript(id_reference, save_path, output_format):
    """Show the content of a transcript by ID."""
    try:
        file_manager = TranscriptFileManager(save_path, output_format)
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


def append_to_transcript(id_reference, save_path, output_format, auto_copy, record_audio_streaming_func, deduplicate_transcript_func):
    """Record new audio and append to existing transcript."""
    try:
        file_manager = TranscriptFileManager(save_path, output_format)
        
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
        new_transcript, session_audio_file, quick_transcript_path, quick_transcript_id = record_audio_streaming_func()
        if not new_transcript:
            print("❌ No new content recorded.")
            return
        
        # Deduplicate the new content
        new_transcript = deduplicate_transcript_func(new_transcript)
        
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
            
            # Remove the duplicate quick transcript file created during recording
            if quick_transcript_path and os.path.exists(quick_transcript_path):
                try:
                    os.remove(quick_transcript_path)
                except Exception as e:
                    print(f"⚠️ Could not remove duplicate transcript: {e}")
            
            # Copy combined content to clipboard if enabled
            if auto_copy:
                combined_content = file_manager.get_transcript_content(id_reference)
                if combined_content:
                    pyperclip.copy(combined_content)
                    print("📋 Combined transcript copied to clipboard.")
        else:
            print(f"❌ Failed to append to transcript {clean_id}")
        
    except Exception as e:
        print(f"❌ Error appending to transcript: {e}")


def summarize_file(path_or_id, save_path, output_format, auto_copy):
    """Summarize and tag a file by path or transcript ID with enhanced UI."""
    import time
    import threading
    from ui_display import show_processing

    try:
        # Initialize summarization service
        summarizer = get_summarizer(notes_folder=save_path)

        # Determine if input is a file path or transcript ID
        file_path = None
        transcript_id = None

        if path_or_id.startswith('-') or path_or_id.isdigit():
            # It's a transcript ID reference
            file_manager = TranscriptFileManager(save_path, output_format)

            try:
                file_path = file_manager.find_transcript(path_or_id)
                transcript_id = file_manager.id_generator.parse_reference_id(path_or_id)
            except ValueError as e:
                print(f"❌ {str(e)}")
                print("💡 Please resolve the naming conflict")
                print("💡 Use 'rec --list' to see all transcripts")
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

            _, ext = os.path.splitext(file_path)
            if ext.lower() not in ['.md', '.txt', '']:
                print(f"⚠️ File type '{ext}' may not be supported. Continuing anyway...")

        file_name = os.path.basename(file_path)

        # Check Ollama availability
        if not summarizer.check_ollama_available():
            print("❌ Ollama not available")
            print("💡 Start Ollama service: 'ollama serve'")
            return

        print(f"🤖 Generating AI summary for: {file_name}")

        # Show processing UI with progress
        start_time = time.time()
        stop_progress = threading.Event()

        def progress_updater():
            """Update UI during AI processing"""
            while not stop_progress.is_set():
                elapsed = time.time() - start_time
                # AI processing is variable, show conservative progress
                progress = min(int((elapsed / 30) * 90), 90)
                show_processing(
                    elapsed,
                    session_id=transcript_id or "external",
                    file_name=file_name,
                    progress=progress
                )
                time.sleep(0.5)

        progress_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_thread.start()

        # Check if this is a transcript file
        is_transcript_file = file_path.startswith(save_path)

        # Summarize the file
        success = summarizer.summarize_file(file_path, copy_to_notes=not is_transcript_file)

        # Stop progress UI
        stop_progress.set()
        progress_thread.join(timeout=1.0)

        processing_time = time.time() - start_time

        if success:
            print(f"✅ AI summary completed in {processing_time:.1f}s")
            print(f"📁 Updated file: {file_path}")

            # Copy to clipboard if enabled
            if auto_copy and is_transcript_file:
                file_manager = TranscriptFileManager(save_path, output_format)
                if transcript_id:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            updated_content = f.read()
                        if updated_content:
                            pyperclip.copy(updated_content)
                            print("📋 Updated transcript copied to clipboard")
                    except Exception as e:
                        print(f"⚠️  Could not copy to clipboard: {e}")
        else:
            print(f"❌ AI summary failed after {processing_time:.1f}s")

    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        import traceback
        traceback.print_exc()


def transcribe_audio_file(audio_path, whisper_model_name):
    """Transcribe a single audio file using the same method as the main transcription."""
    import whisper_engine as whisper
    
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load Whisper model
        whisper_model = whisper.load_model(whisper_model_name)
        
        # Transcribe the audio file
        result = whisper_model.transcribe(str(audio_file), verbose=False)
        transcript_text = result.get('text', '').strip()
        
        return transcript_text if transcript_text else ""
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")


def reprocess_transcript_command(id_reference, save_path, output_format, overwrite_existing, auto_metadata, whisper_model_name):
    """Reprocess all audio files for a specific transcript ID."""
    try:
        file_manager = TranscriptFileManager(save_path, output_format)
        
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
        if auto_metadata:
            try:
                summarizer = get_summarizer()
                if summarizer.check_ollama_available():
                    def summarize_transcript(transcript_text: str) -> dict:
                        """Generate AI summary and metadata."""
                        return summarizer.get_metadata(transcript_text) or {}
                    
                    summarization_callback = summarize_transcript
                else:
                    print("⚠️ Ollama not available - skipping AI summarization")
                    
            except Exception as e:
                print(f"⚠️ AI summarization not available: {str(e)}")
        
        # Transcription callback
        def transcribe_callback(audio_path):
            return transcribe_audio_file(audio_path, whisper_model_name)
        
        # Perform reprocessing
        success, transcript_path, processed_files = file_manager.reprocess_transcript_audio(
            id_reference,
            transcription_callback=transcribe_callback,
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


def reprocess_failed_command(save_path, output_format, auto_metadata, whisper_model_name):
    """Reprocess all orphaned audio files."""
    try:
        file_manager = TranscriptFileManager(save_path, output_format)
        
        # Define summarization callback if AI is enabled
        summarization_callback = None
        if auto_metadata:
            try:
                summarizer = get_summarizer()
                if summarizer.check_ollama_available():
                    def summarize_transcript(transcript_text: str) -> dict:
                        """Generate AI summary and metadata."""
                        return summarizer.get_metadata(transcript_text) or {}
                    
                    summarization_callback = summarize_transcript
                else:
                    print("⚠️ Ollama not available - skipping AI summarization")
                    
            except Exception as e:
                print(f"⚠️ AI summarization not available: {str(e)}")
        
        # Transcription callback
        def transcribe_callback(audio_path):
            return transcribe_audio_file(audio_path, whisper_model_name)
        
        # Perform batch reprocessing
        results = file_manager.reprocess_all_failed_transcripts(
            transcription_callback=transcribe_callback,
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


def list_recovery_sessions(save_path, sample_rate):
    """List available recovery sessions"""
    temp_audio_dir = Path(save_path or tempfile.gettempdir()) / "audio_sessions"
    
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
            duration = file_size / (sample_rate * 2)  # 16-bit mono
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


def recover_session(session_id_or_latest, save_path, output_format, sample_rate, auto_metadata,
                    whisper_model_name, transcribe_session_file_func):
    """Recover and transcribe a specific session"""
    import whisper_engine as whisper
    
    sessions = list_recovery_sessions(save_path, sample_rate)
    
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
        whisper_model = whisper.load_model(whisper_model_name)
        
        # Transcribe session
        transcript = transcribe_session_file_func(session['file'], whisper_model)
        
        if transcript and transcript.strip():
            # Save transcript normally
            file_manager = TranscriptFileManager(save_path, output_format)
            file_path, transcript_id, _ = file_manager.create_new_transcript(transcript.strip(), "recovered_recording")
            
            print(f"✅ Recovery successful!")
            print(f"📄 Transcript {transcript_id} saved: {file_path}")
            
            # Add AI-generated summary and tags (if enabled)
            if auto_metadata:
                print("🤖 Generating summary and tags...")
                summarizer = get_summarizer()
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


def copy_transcript_to_clipboard(id_reference, save_path, output_format):
    """Copy transcript content to clipboard by ID."""
    try:
        file_manager = TranscriptFileManager(save_path, output_format)

        # Find the transcript file
        try:
            file_path = file_manager.find_transcript(id_reference)
        except ValueError as e:
            print(f"❌ {str(e)}")
            print("💡 Please resolve the naming conflict - multiple files have the same ID")
            return

        if not file_path:
            print(f"❌ Transcript with ID '{id_reference}' not found.")
            print("💡 Use 'rec --list' to see available transcripts")
            return

        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading transcript: {e}")
            return

        if content:
            import pyperclip
            pyperclip.copy(content)
            clean_id = file_manager.id_generator.parse_reference_id(id_reference)
            print(f"✅ Transcript {clean_id} copied to clipboard ({len(content)} characters)")
        else:
            print(f"❌ Transcript is empty")

    except Exception as e:
        print(f"❌ Error copying transcript: {e}")


def copy_latest_transcript_to_clipboard(save_path, output_format):
    """Copy most recent transcript to clipboard."""
    try:
        file_manager = TranscriptFileManager(save_path, output_format)
        transcripts = file_manager.list_transcripts_with_audio()

        if not transcripts:
            print("❌ No transcripts found.")
            print("💡 Record your first transcript with 'rec'")
            return

        # Get most recent (first in list, already sorted by date descending)
        # Tuple format: (id, filename, creation_date, audio_count, total_duration)
        latest = transcripts[0]
        latest_id = latest[0]
        latest_filename = latest[1]

        # Find and read the file
        file_path = file_manager.find_transcript(latest_id)
        if not file_path:
            print(f"❌ Could not find latest transcript file")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading transcript: {e}")
            return

        if content:
            import pyperclip
            pyperclip.copy(content)
            print(f"✅ Latest transcript ({latest_id}) copied to clipboard")
            print(f"   File: {latest_filename}")
            print(f"   Size: {len(content)} characters")
        else:
            print(f"❌ Latest transcript is empty")

    except Exception as e:
        print(f"❌ Error copying latest transcript: {e}")


def open_transcript_file(id_reference, save_path, output_format, obsidian_enabled, obsidian_vault_path):
    """Open transcript file by ID in appropriate application."""
    import sys
    import subprocess

    try:
        file_manager = TranscriptFileManager(save_path, output_format)

        # Find the transcript file
        try:
            file_path = file_manager.find_transcript(id_reference)
        except ValueError as e:
            print(f"❌ {str(e)}")
            print("💡 Please resolve the naming conflict - multiple files have the same ID")
            return

        if not file_path:
            print(f"❌ Transcript with ID '{id_reference}' not found.")
            print("💡 Use 'rec --list' to see available transcripts")
            return

        clean_id = file_manager.id_generator.parse_reference_id(id_reference)
        filename = os.path.basename(file_path)

        # Try Obsidian first if enabled and file is markdown in vault
        if file_path.endswith('.md') and obsidian_enabled and obsidian_vault_path:
            from obsidian_utils import open_in_obsidian, is_path_in_vault

            if is_path_in_vault(file_path, obsidian_vault_path):
                if open_in_obsidian(file_path, obsidian_vault_path):
                    print(f"📂 Opened transcript {clean_id} in Obsidian")
                    print(f"   File: {filename}")
                    return

        # Fall back to default application
        if sys.platform == "darwin":
            subprocess.run(["open", file_path])
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", file_path])
        elif sys.platform == "win32":
            subprocess.run(["explorer", file_path])
        else:
            print(f"📁 File location: {file_path}")
            return

        print(f"📂 Opened transcript {clean_id}")
        print(f"   File: {filename}")

    except Exception as e:
        print(f"❌ Error opening transcript: {e}")


def show_commands_list():
    """Display categorized list of all available commands."""
    print("\n" + "="*60)
    print("REJOICE - Available Commands")
    print("="*60)

    # Main Operations
    print("\n📝 MAIN OPERATIONS")
    print("-" * 60)
    print("  rec                    Start new recording")
    print("  rec ID                 Append to existing transcript")
    print("  rec -l, --list         List all transcripts")
    print("  rec -v, --view ID      Show transcript content")
    print("  rec -o, --open ID      Open transcript in editor/Obsidian")
    print("  rec -s, --settings     Open settings menu")

    # Transcription
    print("\n🎵 TRANSCRIPTION")
    print("-" * 60)
    print("  rec --transcribe FILE  Transcribe audio file (WAV)")
    print("  rec -r, --recover [ID] Recover interrupted session")
    print("  rec -ls                List recoverable sessions")
    print("  rec --reprocess ID     Re-transcribe all audio for ID")
    print("  rec --reprocess-failed Process orphaned audio files")

    # AI Operations
    print("\n🤖 AI OPERATIONS")
    print("-" * 60)
    print("  rec -g, --genai ID     Generate AI summary and tags")
    print("  rec -g FILE            Summarize any text file")

    # Clipboard Operations
    print("\n📋 CLIPBOARD OPERATIONS")
    print("-" * 60)
    print("  rec -c, --copy ID      Copy transcript to clipboard")
    print("  rec --copy-latest      Copy most recent transcript")

    # File Operations
    print("\n📁 FILE OPERATIONS")
    print("-" * 60)
    print("  rec --open-folder      Open transcripts folder")
    print("  rec --audio ID         Show audio files for transcript")

    # Recording Modifiers
    print("\n⚙️  RECORDING MODIFIERS")
    print("-" * 60)
    print("  rec --auto-copy        Auto-copy after recording")
    print("  rec --no-copy          Don't auto-copy")
    print("  rec --auto-open        Auto-open file after recording")
    print("  rec --no-open          Don't auto-open")
    print("  rec --metadata         Generate AI metadata after recording")
    print("  rec --no-metadata      Skip AI metadata")
    print("  rec --device N         Use specific microphone device")
    print("  rec -ts, --timestamps  Include timestamps in output")

    # Utility
    print("\n🔧 UTILITY")
    print("-" * 60)
    print("  rec --commands         Show this command list")
    print("  rec --debug            Enable debug output")

    print("\n" + "="*60)
    print("💡 TIP: Most commands support both short (-l) and long (--list) forms")
    print("💡 TIP: Use 'rec -s' to configure default behaviors")
    print("="*60 + "\n")


def transcribe_arbitrary_audio(audio_path, save_path, output_format, whisper_model_name,
                                auto_metadata, timestamps=False):
    """
    Transcribe an arbitrary audio file and save as new transcript.

    Uses enhanced UI display pattern matching live recording.
    Supports WAV format only (like session recovery).
    """
    import whisper_engine as whisper
    import time
    import threading
    import wave
    from pathlib import Path
    from ui_display import show_transcribing, show_processing, show_complete

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return None

    # Validate it's a WAV file
    if audio_file.suffix.lower() not in ['.wav', '.wave']:
        print(f"❌ Only WAV audio files are supported")
        print(f"💡 Convert your audio to WAV format first")
        return None

    print(f"🎵 Transcribing audio file: {audio_file.name}")

    try:
        # Get audio duration for progress estimation
        with wave.open(str(audio_file), 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration_seconds = frames / float(rate)

        print(f"📊 Audio duration: {duration_seconds:.1f}s")

        # Phase 1: Transcription with UI
        print("🔄 Loading Whisper model...")
        whisper_model = whisper.load_model(whisper_model_name)

        show_transcribing(0, session_id=audio_file.stem)

        start_time = time.time()
        stop_progress = threading.Event()

        # Estimate transcription time (same multipliers as main recording)
        model_multipliers = {
            'tiny': 0.15, 'base': 0.2, 'small': 0.3,
            'medium': 0.5, 'large': 0.7, 'large-v2': 0.7, 'large-v3': 0.7
        }
        multiplier = model_multipliers.get(whisper_model_name.lower(), 0.3)
        estimated_time = max(duration_seconds * multiplier, 5.0)

        def progress_updater():
            """Update transcription progress UI"""
            while not stop_progress.is_set():
                elapsed = time.time() - start_time
                progress = min(int((elapsed / estimated_time) * 95), 95)
                show_transcribing(elapsed, session_id=audio_file.stem, progress=progress)
                time.sleep(0.5)

        progress_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_thread.start()

        # Transcribe
        result = whisper_model.transcribe(str(audio_file), fp16=False)

        stop_progress.set()
        progress_thread.join(timeout=1.0)

        # Format transcript with timestamps if requested
        if timestamps and 'segments' in result:
            # Local import to avoid circular dependency
            from transcribe import format_transcript_with_timestamps
            transcript_text = format_transcript_with_timestamps(result['segments'])
        else:
            transcript_text = result.get('text', '').strip()

        if not transcript_text:
            print("❌ No speech detected in audio file")
            return None

        transcription_time = time.time() - start_time
        print(f"✅ Transcription completed in {transcription_time:.1f}s")

        # Phase 2: Save transcript
        file_manager = TranscriptFileManager(save_path, output_format)
        file_path, transcript_id, stored_audio_path = file_manager.create_new_transcript(
            transcript_text=transcript_text,
            generated_filename="transcript",
            session_audio_file=audio_file
        )

        word_count = len(transcript_text.split())

        # Phase 3: AI metadata (optional, with UI)
        if auto_metadata and transcript_text.strip():
            show_processing(0, session_id=transcript_id, file_name=os.path.basename(file_path))

            try:
                summarizer = get_summarizer()
                if summarizer.check_ollama_available():
                    ai_start = time.time()
                    stop_ai_progress = threading.Event()

                    def ai_progress_updater():
                        """Update AI processing progress UI"""
                        while not stop_ai_progress.is_set():
                            elapsed = time.time() - ai_start
                            progress = min(int((elapsed / 30) * 90), 90)
                            show_processing(elapsed, session_id=transcript_id,
                                          file_name=os.path.basename(file_path),
                                          progress=progress)
                            time.sleep(0.5)

                    ai_thread = threading.Thread(target=ai_progress_updater, daemon=True)
                    ai_thread.start()

                    success = summarizer.summarize_file(file_path, copy_to_notes=False)

                    stop_ai_progress.set()
                    ai_thread.join(timeout=1.0)

                    if success:
                        print("✅ AI metadata generated successfully")
                else:
                    print("⚠️  Ollama not available - skipping AI metadata")
            except Exception as e:
                print(f"⚠️  AI metadata failed: {e}")

        # Phase 4: Show completion UI (matches recording flow)
        show_complete(
            session_id=transcript_id,
            file_name=os.path.basename(file_path),
            file_path=file_path,
            duration_seconds=duration_seconds,
            word_count=word_count
        )

        return {
            'transcript_id': transcript_id,
            'file_path': file_path,
            'transcript_text': transcript_text,
            'word_count': word_count
        }

    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None
