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

def validate_save_path(path: str) -> tuple:
    """
    Check if path is in cloud-synced directory and warn user.

    This is a critical privacy check to prevent transcripts from being
    automatically uploaded to cloud services, maintaining the "un-snoopable" promise.

    Args:
        path: The save path to validate

    Returns:
        tuple: (is_safe: bool, warning_message: str)
    """
    # Known cloud service indicators in paths
    cloud_indicators = {
        'icloud': 'iCloud',
        'dropbox': 'Dropbox',
        'google drive': 'Google Drive',
        'googledrive': 'Google Drive',
        'onedrive': 'OneDrive',
        'box sync': 'Box',
        'sync': 'Sync.com',
        'cloudstation': 'Synology CloudStation',
        'mega': 'MEGA',
        'pcloud': 'pCloud',
        'tresorit': 'Tresorit'
    }

    # Normalize path for checking
    normalized_path = path.replace('\\', '/').lower()

    # Check for cloud indicators
    detected_service = None
    for indicator, service_name in cloud_indicators.items():
        if indicator in normalized_path:
            detected_service = service_name
            break

    if detected_service:
        warning = f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    ⚠️  PRIVACY WARNING ⚠️                         ║
╚═══════════════════════════════════════════════════════════════════╝

The selected path appears to be in a {detected_service} folder:
  {path}

This means your transcripts will be:
  • Uploaded to cloud servers automatically
  • Potentially indexed and analyzed by the cloud provider
  • Accessible from all your synced devices
  • Subject to the cloud provider's privacy policy and data handling
  • Possibly scanned for content moderation or other purposes

⚠️  This VIOLATES Rejoice's core promise of "un-snoopable" transcription!

📁 Recommended local-only directories:
  • ~/Documents/Transcripts
  • ~/Desktop/Transcripts
  • ~/Local/Transcripts
  • /usr/local/share/transcripts

💡 Tip: Create a dedicated local folder for maximum privacy.
"""
        return False, warning

    return True, ""

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

    # Security: Ensure .env has secure permissions (owner read/write only)
    try:
        import stat
        os.chmod(env_path, stat.S_IRUSR | stat.S_IWUSR)  # 600 permissions
    except Exception:
        # Non-critical - just skip if permission setting fails
        pass

    # Also update the current process environment
    os.environ[key] = value

def display_settings_overview():
    """Display comprehensive settings overview with navigation map"""
    print("\n" + "=" * 70)
    print("📋 CURRENT CONFIGURATION OVERVIEW")
    print("=" * 70)

    # Read current configuration
    config = {}
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key] = value.strip("'\"")

    # 🎯 CORE - Commonly Adjusted Settings
    print("\n🎯 CORE (Commonly Adjusted)")
    print(f"   Save Path:          {config.get('SAVE_PATH', 'N/A')}")
    print(f"   Output Format:      {config.get('OUTPUT_FORMAT', 'N/A')}")
    print(f"   Whisper Model:      {config.get('WHISPER_MODEL', 'N/A')}")
    print(f"   Language:           {config.get('WHISPER_LANGUAGE', 'N/A')}")
    print(f"   Ollama Model:       {config.get('OLLAMA_MODEL', 'N/A')}")
    mic_device = config.get('DEFAULT_MIC_DEVICE', '-1')
    print(f"   Microphone:         {mic_device if mic_device != '-1' else 'System Default'}")

    # 🎨 CASUAL - Quality of Life
    print("\n🎨 CASUAL (Quality of Life)")
    auto_metadata = config.get('AUTO_METADATA', 'false') == 'true'
    print(f"   Auto Metadata:      {'✅ Yes' if auto_metadata else '❌ No'}")
    command = config.get('COMMAND_NAME', 'rec')
    print(f"   Command Name:       {command}")
    auto_copy = config.get('AUTO_COPY', 'false') == 'true'
    auto_open = config.get('AUTO_OPEN', 'false') == 'true'
    print(f"   Auto Copy:          {'✅ Yes' if auto_copy else '❌ No'}")
    print(f"   Auto Open:          {'✅ Yes' if auto_open else '❌ No'}")

    # Obsidian Integration
    obsidian_enabled = config.get('OBSIDIAN_ENABLED', 'false') == 'true'
    obsidian_vault = config.get('OBSIDIAN_VAULT_PATH', '')
    if obsidian_enabled and obsidian_vault:
        vault_name = os.path.basename(obsidian_vault)
        print(f"   Obsidian Vault:     ✅ {vault_name}")
    else:
        print(f"   Obsidian Vault:     ❌ Not configured")

    # ⚙️ ADVANCED - Technical Settings
    print("\n⚙️ ADVANCED (Technical)")
    buffer = int(config.get('STREAMING_BUFFER_SIZE_SECONDS', '0'))
    buffer_str = f"{buffer//60}m {buffer%60}s" if buffer >= 60 else f"{buffer}s"
    print(f"   Buffer Size:        {buffer_str} (rolling audio buffer)")
    min_seg = config.get('STREAMING_MIN_SEGMENT_DURATION', 'N/A')
    target_seg = config.get('STREAMING_TARGET_SEGMENT_DURATION', 'N/A')
    max_seg = config.get('STREAMING_MAX_SEGMENT_DURATION', 'N/A')
    print(f"   Segments:           {min_seg}s-{target_seg}s-{max_seg}s (min-target-max chunks)")
    print(f"   Ollama API:         {config.get('OLLAMA_API_URL', 'N/A')}")
    timeout = int(config.get('OLLAMA_TIMEOUT', '0'))
    timeout_str = f"{timeout//60}m {timeout%60}s" if timeout >= 60 else f"{timeout}s"
    print(f"   Ollama Timeout:     {timeout_str}")
    max_length = config.get('OLLAMA_MAX_CONTENT_LENGTH', 'N/A')
    print(f"   Max AI Content:     {int(max_length):,} chars" if max_length.isdigit() else f"   Max AI Content:     {max_length}")

    print("\n" + "=" * 70)
    print("💡 HOW TO CHANGE SETTINGS")
    print("=" * 70)
    print(f"   Run: {command} --settings")
    print("\n   Then navigate to:")
    print("   • Option 1 - 🎯 Core settings (path, format, models, mic)")
    print("   • Option 2 - 🎨 Casual settings (auto-actions, command name)")
    print("   • Option 3 - ⚙️  Advanced settings (streaming, Ollama config)")
    print("=" * 70)

def settings_menu():
    """Interactive settings menu with categories"""
    try:
        print("\n⚙️  Rejoice Slim Settings")
        print("   Record. Jot. Voice.")
        print("─" * 50)

        # Display overview on first load
        display_settings_overview()

        while True:
            print("\n📋 Settings Categories:")
            print("  1. 🎯 Core (Path, format, models, mic)")
            print("  2. 🎨 Casual (Auto-actions, command name)")
            print("  3. ⚙️  Advanced (Streaming, Ollama config)")
            print("  4. 🗑️  Uninstall (Remove aliases, venv, and config)")
            print("  5. 📋 Show Overview (Display current configuration)")
            print("  6. 🚪 Exit")

            choice = input("\n👉 Choose a category (1-6): ").strip()

            if choice == "1":
                core_settings()
            elif choice == "2":
                casual_settings()
            elif choice == "3":
                advanced_settings()
            elif choice == "4":
                uninstall_settings()
            elif choice == "5":
                display_settings_overview()
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

def core_settings():
    """Core settings submenu - commonly adjusted settings"""
    while True:
        # Get current values
        SAVE_PATH = os.getenv("SAVE_PATH")
        OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "md")
        WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
        WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
        DEFAULT_MIC_DEVICE = int(os.getenv("DEFAULT_MIC_DEVICE", "-1"))

        print(f"\n🎯 Core Settings")
        print("─" * 18)
        print(f"Save Path:      {SAVE_PATH}")
        print(f"Output Format:  {OUTPUT_FORMAT}")
        print(f"Whisper Model:  {WHISPER_MODEL}")
        print(f"Language:       {WHISPER_LANGUAGE}")
        print(f"Ollama Model:   {OLLAMA_MODEL}")
        print(f"Microphone:     {DEFAULT_MIC_DEVICE if DEFAULT_MIC_DEVICE != -1 else 'System Default'}")
        print(f"\n1. Change Save Path")
        print(f"2. Change Output Format")
        print(f"3. Change Whisper Model")
        print(f"4. Change Language")
        print(f"5. Change Ollama Model")
        print(f"6. Change Microphone Device")
        print(f"7. ← Back to Main Menu")

        choice = input("\n👉 Choose option (1-7): ").strip()

        if choice == "1":
            new_path = input(f"Enter new save path [{SAVE_PATH}]: ").strip()
            if new_path:
                # Expand user path (~/... becomes /Users/username/...)
                new_path = os.path.expanduser(new_path)

                # Security: Check if path is cloud-synced
                is_safe, warning = validate_save_path(new_path)
                if not is_safe:
                    print(warning)
                    confirm = input("\n⚠️  Continue with this cloud-synced path anyway? [y/N]: ").strip().lower()
                    if confirm not in ['y', 'yes']:
                        print("✅ Save path change cancelled for security")
                        continue

                # Create directory if it doesn't exist
                os.makedirs(new_path, exist_ok=True)
                update_env_setting("SAVE_PATH", new_path)
                print(f"✅ Save path changed to: {new_path}")
                if not is_safe:
                    print("⚠️  Remember: Your transcripts will sync to the cloud!")
                print("⚠️ Restart the script to use the new path")

        elif choice == "2":
            format_choice = input("Choose output format (md/txt): ").strip().lower()
            if format_choice in ["md", "txt"]:
                update_env_setting("OUTPUT_FORMAT", format_choice)
                print(f"✅ Output format changed to: {format_choice}")
                print("⚠️ Restart the script to use the new format")

        elif choice == "3":
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
                    print(f"✅ Whisper model changed to: {new_model}")
                    print("✅ Model downloaded and ready to use")
                except Exception as e:
                    print(f"❌ Failed to download model: {e}")
                    print("⚠️ Model setting not updated")

        elif choice == "4":
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

        elif choice == "5":
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

        elif choice == "6":
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

        elif choice == "7":
            break
        else:
            print("❌ Invalid choice. Please select 1-7.")

def casual_settings():
    """Casual settings submenu - quality of life settings"""
    while True:
        # Get current values
        AUTO_METADATA = os.getenv("AUTO_METADATA", "false").lower() == "true"
        COMMAND_NAME = os.getenv("COMMAND_NAME", "rec")
        AUTO_COPY = os.getenv("AUTO_COPY", "false").lower() == "true"
        AUTO_OPEN = os.getenv("AUTO_OPEN", "false").lower() == "true"
        OBSIDIAN_ENABLED = os.getenv("OBSIDIAN_ENABLED", "false").lower() == "true"
        OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "")

        # Display Obsidian vault status
        if OBSIDIAN_ENABLED and OBSIDIAN_VAULT_PATH:
            vault_name = os.path.basename(OBSIDIAN_VAULT_PATH)
            obsidian_status = f"✅ {vault_name}"
        else:
            obsidian_status = "❌ Not configured"

        print(f"\n🎨 Casual Settings (Quality of Life)")
        print("─" * 38)
        print(f"Auto Metadata:      {'Yes' if AUTO_METADATA else 'No'}")
        print(f"Command Name:       {COMMAND_NAME}")
        print(f"Auto Copy:          {'Yes' if AUTO_COPY else 'No'}")
        print(f"Auto Open:          {'Yes' if AUTO_OPEN else 'No'}")
        print(f"Obsidian Vault:     {obsidian_status}")
        print(f"\n1. Toggle Auto Metadata")
        print(f"2. Change Command Name")
        print(f"3. Toggle Auto Copy")
        print(f"4. Toggle Auto Open")
        print(f"5. Configure Obsidian Integration")
        print(f"6. ← Back to Main Menu")

        choice = input("\n👉 Choose option (1-6): ").strip()

        if choice == "1":
            new_setting = input("Auto generate AI metadata? (y/n): ").lower()
            if new_setting in ['y', 'n']:
                update_env_setting("AUTO_METADATA", 'true' if new_setting == 'y' else 'false')
                print(f"✅ Auto metadata changed to: {'Yes' if new_setting == 'y' else 'No'}")
                print("⚠️ Restart the script to use the new setting")

        elif choice == "2":
            print(f"\nCurrent command: {COMMAND_NAME}")
            print("Examples: rec, record, transcribe, voice, tr, etc.")
            print("Choose something that won't conflict with existing commands.")

            new_command = input("Enter new command name: ").strip()

            if new_command and new_command != COMMAND_NAME:
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

                        print(f"✅ Command changed from '{COMMAND_NAME}' to '{new_command}'")
                        print(f"🔄 Please restart your terminal or run 'source ~/.zshrc' to use the new command")
                        print(f"💡 Your old command '{COMMAND_NAME}' will no longer work")

                    else:
                        print("❌ Could not find ~/.zshrc file")

                except Exception as e:
                    print(f"❌ Error updating alias: {e}")
                    print("💡 You may need to manually update your ~/.zshrc file")
            elif new_command == COMMAND_NAME:
                print("ℹ️  Command name is already set to that value")
            else:
                print("❌ Invalid command name")

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
            # Configure Obsidian Integration
            from obsidian_utils import configure_obsidian_integration

            # Get save path to use for vault selection
            SAVE_PATH = os.getenv("SAVE_PATH", "")
            if not SAVE_PATH:
                print("❌ Save path not configured. Please configure save path first.")
                continue

            obsidian_enabled, obsidian_vault_path = configure_obsidian_integration(SAVE_PATH)

            # Update .env settings
            update_env_setting("OBSIDIAN_ENABLED", 'true' if obsidian_enabled else 'false')
            update_env_setting("OBSIDIAN_VAULT_PATH", obsidian_vault_path)

            if obsidian_enabled:
                vault_name = os.path.basename(obsidian_vault_path)
                print(f"\n✅ Obsidian integration enabled for vault: {vault_name}")
            else:
                print(f"\n✅ Obsidian integration disabled")
            print("⚠️ Restart the script to use the new setting")

        elif choice == "6":
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

def advanced_settings():
    """Advanced settings submenu - technical settings"""
    while True:
        # Read current advanced settings
        current_buffer = int(os.getenv('STREAMING_BUFFER_SIZE_SECONDS', '300'))
        current_min = int(os.getenv('STREAMING_MIN_SEGMENT_DURATION', '30'))
        current_target = int(os.getenv('STREAMING_TARGET_SEGMENT_DURATION', '60'))
        current_max = int(os.getenv('STREAMING_MAX_SEGMENT_DURATION', '90'))
        current_api_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
        current_timeout = int(os.getenv('OLLAMA_TIMEOUT', '180'))
        current_max_length = int(os.getenv('OLLAMA_MAX_CONTENT_LENGTH', '32000'))

        print(f"\n⚙️  Advanced Settings (Technical)")
        print("─" * 35)
        buffer_str = f"{current_buffer//60}m {current_buffer%60}s" if current_buffer >= 60 else f"{current_buffer}s"
        print(f"Buffer Size:        {buffer_str} (rolling audio buffer)")
        print(f"Segments:           {current_min}s-{current_target}s-{current_max}s (min-target-max chunks)")
        print(f"Ollama API:         {current_api_url}")
        timeout_str = f"{current_timeout//60}m {current_timeout%60}s" if current_timeout >= 60 else f"{current_timeout}s"
        print(f"Ollama Timeout:     {timeout_str}")
        print(f"Max AI Content:     {current_max_length:,} chars")
        print(f"\n1. Configure Streaming Buffer Size")
        print(f"2. Configure Streaming Segment Durations")
        print(f"3. Change Ollama API URL")
        print(f"4. Change Ollama Timeout")
        print(f"5. Change Max AI Content Length")
        print(f"6. ← Back to Main Menu")

        choice = input("\n👉 Choose option (1-6): ").strip()

        if choice == "1":
            print(f"\nCurrent buffer size: {current_buffer} seconds ({current_buffer//60}m {current_buffer%60}s)")
            print("How much audio to keep in memory for context:")
            print("  • 300s (5m)   - Short sessions, low memory")
            print("  • 600s (10m)  - Balanced (recommended)")
            print("  • 900s (15m)  - Long sessions, high quality")

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
            print("Control how audio is broken into chunks:")
            print("  • Min: Don't transcribe until at least this much speech")
            print("  • Target: Look for natural pauses around this duration")
            print("  • Max: Force break at this point (prevents memory issues)")

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
            print("\n⚠️  SECURITY NOTICE: Ollama API URL must be localhost-only")
            print("Rejoice only supports local Ollama instances to protect your privacy.")
            print("\nValid examples:")
            print("  • http://localhost:11434/api/generate")
            print("  • http://127.0.0.1:11434/api/generate")

            new_url = input("\nEnter Ollama API URL: ").strip()
            if new_url:
                # Validate the URL is localhost-only before saving
                from urllib.parse import urlparse
                try:
                    parsed = urlparse(new_url)
                    hostname = parsed.hostname
                    valid_hosts = ['localhost', '127.0.0.1', '::1', '0.0.0.0']

                    if hostname is None or hostname.lower() not in valid_hosts:
                        print(f"\n❌ REJECTED: URL must point to localhost, got: {hostname or 'invalid'}")
                        print("   Rejoice will not send your transcripts to remote servers.")
                        print("   This protects your privacy and maintains the 'un-snoopable' promise.")
                        continue

                    update_env_setting("OLLAMA_API_URL", new_url)
                    print(f"✅ Ollama API URL changed to: {new_url}")
                    print("⚠️ Restart the script to use the new URL")
                except Exception as e:
                    print(f"❌ Invalid URL format: {e}")
                    print("   Please use format: http://localhost:PORT/api/generate")

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
            print("  • 8,000   - Conservative")
            print("  • 32,000  - Balanced (recommended)")
            print("  • 64,000  - For powerful setups")

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

