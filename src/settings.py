# src/settings.py
"""
Settings menu system for managing transcriber configuration.
Extracted from transcribe.py for better modularity.
"""

import os
import sys
import shutil

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
    # Get current values
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
    
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
                    import whisper_engine as whisper
                    whisper.load_model(new_model)
                    update_env_setting("WHISPER_MODEL", new_model)
                    WHISPER_MODEL = new_model
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
                WHISPER_LANGUAGE = new_language
                print(f"✅ Whisper language changed to: {new_language}")
                print("⚠️ Restart the script to use the new language")
        
        elif choice == "3":
            break
        else:
            print("❌ Invalid choice. Please select 1-3.")

def output_settings():
    """Output settings submenu"""
    # Get current values
    OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "md")
    SAVE_PATH = os.getenv("SAVE_PATH")
    AUTO_COPY = os.getenv("AUTO_COPY", "false").lower() == "true"
    AUTO_OPEN = os.getenv("AUTO_OPEN", "false").lower() == "true"
    AUTO_METADATA = os.getenv("AUTO_METADATA", "false").lower() == "true"
    AUTO_CLEANUP_AUDIO = os.getenv("AUTO_CLEANUP_AUDIO", "true").lower() == "true"
    
    while True:
        print(f"\n📁 Output Settings")
        print("─" * 20)
        print(f"Current Format: {OUTPUT_FORMAT}")
        print(f"Current Save Path: {SAVE_PATH}")
        print(f"Auto Copy: {'Yes' if AUTO_COPY else 'No'}")
        print(f"Auto Open: {'Yes' if AUTO_OPEN else 'No'}")
        print(f"Auto Metadata: {'Yes' if AUTO_METADATA else 'No'}")
        print(f"Auto Cleanup Audio: {'Yes' if AUTO_CLEANUP_AUDIO else 'No'}")
        print(f"\n1. Change Output Format")
        print(f"2. Change Save Path")
        print(f"3. Toggle Auto Copy")
        print(f"4. Toggle Auto Open")
        print(f"5. Toggle Auto Metadata")
        print(f"6. Toggle Auto Cleanup Audio")
        print(f"7. ← Back to Main Menu")
        
        choice = input("\n👉 Choose option (1-7): ").strip()
        
        if choice == "1":
            format_choice = input("Choose output format (md/txt): ").strip().lower()
            if format_choice in ["md", "txt"]:
                update_env_setting("OUTPUT_FORMAT", format_choice)
                OUTPUT_FORMAT = format_choice
                print(f"✅ Output format changed to: {format_choice}")
                print("⚠️ Restart the script to use the new format")
        
        elif choice == "2":
            new_path = input(f"Enter new save path [{SAVE_PATH}]: ").strip()
            if new_path:
                os.makedirs(new_path, exist_ok=True)
                update_env_setting("SAVE_PATH", new_path)
                SAVE_PATH = new_path
                print(f"✅ Save path changed to: {new_path}")
                print("⚠️ Restart the script to use the new path")
        
        elif choice == "3":
            new_setting = input("Auto copy to clipboard? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_COPY", 'true' if new_setting == 'y' else 'false')
                AUTO_COPY = (new_setting == 'y')
                print(f"✅ Auto copy changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "4":
            new_setting = input("Auto open file? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_OPEN", 'true' if new_setting == 'y' else 'false')
                AUTO_OPEN = (new_setting == 'y')
                print(f"✅ Auto open changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "5":
            new_setting = input("Auto generate AI metadata? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_METADATA", 'true' if new_setting == 'y' else 'false')
                AUTO_METADATA = (new_setting == 'y')
                print(f"✅ Auto metadata changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "6":
            new_setting = input("Auto cleanup audio files after transcription? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_CLEANUP_AUDIO", 'true' if new_setting == 'y' else 'false')
                AUTO_CLEANUP_AUDIO = (new_setting == 'y')
                print(f"✅ Auto cleanup audio changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("💡 When enabled, audio files are deleted after full transcription")
                print("💡 When disabled, audio files are kept for reprocessing")
                print("⚠️ Restart the script to use the new setting")
        
        elif choice == "7":
            break
        else:
            print("❌ Invalid choice. Please select 1-7.")

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
        
        choice = input("\n👉 Choose option (1-6): ").strip()
        
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
    # Get current value
    DEFAULT_MIC_DEVICE = int(os.getenv("DEFAULT_MIC_DEVICE", "-1"))
    
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
                        DEFAULT_MIC_DEVICE = device_num
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
        # SILENCE_DURATION_SECONDS removed - it's not actually used in the recording loop
        print(f"Streaming Buffer: {current_buffer}s ({current_buffer//60}m {current_buffer%60}s)")
        print(f"Streaming Segments: {current_min}s-{current_target}s-{current_max}s (min-target-max)")
        print(f"Streaming Verbose: {'Yes' if current_verbose else 'No'}")
        print(f"\n1. Configure Streaming Buffer Size")
        print(f"2. Configure Streaming Segment Durations")
        print(f"3. Toggle Streaming Verbose Mode")
        print(f"4. ← Back to Main Menu")

        choice = input("\n👉 Choose option (1-4): ").strip()

        if choice == "1":
            print(f"\nCurrent buffer size: {current_buffer} seconds ({current_buffer//60}m {current_buffer%60}s)")
            print("Recommended buffer sizes:")
            print("  • 300s (5m)   - Short sessions, low memory")
            print("  • 900s (15m)  - Balanced (default, recommended)")
            print("  • 1200s (20m) - Long sessions, high quality")
            
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
        
        elif choice == "2":
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

        elif choice == "3":
            new_verbose = input("Enable streaming verbose mode? (y/n): ").lower()
            if new_verbose in ['y', 'n']:
                update_env_setting("STREAMING_VERBOSE", 'true' if new_verbose == 'y' else 'false')
                print(f"✅ Streaming verbose mode changed to: {'Yes' if new_verbose == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")

        elif choice == "4":
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

def command_settings():
    """Command settings submenu"""
    while True:
        # Read current command name
        current_command = os.getenv('COMMAND_NAME', 'rec')
        
        print(f"\n🔧 Command Settings")
        print("─" * 20)
        print(f"Current command: {current_command}")
        print(f"\n📖 Available Commands:")
        print(f"  {current_command}                    - Start recording")
        print(f"  {current_command} -s, --settings     - Open settings menu")
        print(f"  {current_command} -l, --list         - List all transcripts")
        print(f"  {current_command} -v ID, --view ID   - View transcript by ID")
        print(f"  {current_command} -g ID, --genai ID  - Generate AI summary/tags")
        print(f"  {current_command} --audio ID         - Show audio files for ID")
        print(f"  {current_command} --reprocess ID     - Reprocess transcript")
        print(f"  {current_command} --reprocess-failed - Reprocess all failed")
        print(f"  {current_command} -o, --open-folder  - Open transcripts folder")
        print(f"  {current_command} -r, --recover      - Recover interrupted session")
        print(f"  {current_command} -ls, --list-sessions - List recovery sessions")
        print(f"  {current_command} --verbose          - Enable verbose logging")
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

