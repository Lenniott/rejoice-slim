# configure.py

import os
import shutil

def display_config_summary():
    """Display a comprehensive configuration summary with navigation guide"""
    print("\n" + "=" * 70)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 70)

    # Read current configuration
    env_path = ".env"
    config = {}
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
    print("💡 HOW TO CHANGE SETTINGS LATER")
    print("=" * 70)
    print(f"\n   Run: {command} --settings")
    print("\n   Then navigate to:")
    print("   • Option 1 - 🎯 Core settings (path, format, models, mic)")
    print("   • Option 2 - 🎨 Casual settings (auto-actions, command name)")
    print("   • Option 3 - ⚙️  Advanced settings (streaming, Ollama config)")
    print("=" * 70 + "\n")

def main():
    print("--- 🎙️ Welcome to Rejoice Slim Setup ---")
    print("Record. Jot. Voice.")
    print("Let's configure your settings. Press Enter to accept defaults.\n")
    
    # Check if Ollama is installed
    ollama_installed = shutil.which('ollama') is not None
    
    # Choose installation mode
    print("Choose installation type:")
    print("1) Basic (recommended) - Quick setup with sensible defaults")
    print("2) Detailed - Configure all advanced settings")
    install_mode = ""
    while install_mode not in ["1", "2"]:
        install_mode = input("Enter your choice (1 or 2) [default: 1]: ").strip() or "1"
    
    is_basic_mode = install_mode == "1"
    
    if is_basic_mode:
        print("✅ Basic mode selected - using sensible defaults for advanced settings")
    else:
        print("✅ Detailed mode selected - you'll configure all settings")
    
    if not ollama_installed:
        print("⚠️  Ollama not detected - AI features (smart filenames, summaries) will be disabled")
        print("   You can install Ollama later from https://ollama.ai to enable these features")
    
    print()

    # 1. Get the recording command (alias)
    rec_command = input("\nEnter the command you'll use to start recording [default: rec]: ") or "rec"

    # 2. Get the directory to save transcripts
    default_path = os.path.join(os.path.expanduser("~"), "Documents", "transcripts")
    save_path = input(f"\n\nEnter the full path to save transcripts [default: {default_path}]: ") or default_path
    os.makedirs(save_path, exist_ok=True)
    print(f"✅ Transcripts will be saved in: {save_path}")

    # 2a. Configure Obsidian integration
    import sys
    sys.path.insert(0, 'src')
    from obsidian_utils import configure_obsidian_integration

    obsidian_enabled, obsidian_vault_path = configure_obsidian_integration(save_path)

    # 3. Get the default output format
    if obsidian_enabled:
        # Obsidian requires markdown format
        output_format = "md"
        print(f"\n✓ Output format set to 'md' (required for Obsidian integration)")
    else:
        output_format = ""
        while output_format not in ["md", "txt"]:
            output_format = input("\n\nChoose the default output format (md/txt) [default: md]: ") or "md"

    # 4. Choose Whisper model
    models = ["tiny", "base", "small", "medium", "large"]
    model_choice = ""
    while model_choice not in models:
        model_choice = input(f"\n\nChoose a Whisper model {models} [default: small]: ") or "small"
    print("Downloading the Whisper model now. This might take a moment...")
    # This part triggers the download during setup
    import sys
    sys.path.insert(0, 'src')
    import whisper_engine as whisper
    whisper.load_model(model_choice)
    print("✅ Model downloaded successfully.")

    # 5. Get Ollama model for naming (only if Ollama is installed)
    if ollama_installed:
        ollama_model = input("\n\nEnter the Ollama model to use for file naming (e.g., gemma3:4b) [default: gemma3:4b]: ") or "gemma3:4b"

        # Additional Ollama settings for advanced mode
        if not is_basic_mode:
            print("\n--- Advanced Ollama Settings ---")
            ollama_api_url = input("Ollama API URL [default: http://localhost:11434/api/generate]: ") or "http://localhost:11434/api/generate"

            print("\nOllama timeout (how long to wait for AI responses):")
            print("  • 60s  - Fast models (gemma3:270m, qwen3:0.6b)")
            print("  • 180s - Medium models (gemma3:4b, llama3)")
            print("  • 300s - Large models (llama3:70b)")
            ollama_timeout = input("Timeout in seconds (30-600) [default: 180]: ") or "180"

            print("\nMax content length (how much text to send to AI):")
            print("  • 8,000   - Conservative")
            print("  • 32,000  - Balanced (recommended)")
            print("  • 64,000  - For powerful setups")
            ollama_max_length = input("Max content length (1000-200000) [default: 32000]: ") or "32000"
        else:
            ollama_api_url = "http://localhost:11434/api/generate"
            ollama_timeout = "180"
            ollama_max_length = "32000"
    else:
        ollama_model = "gemma3:4b"  # Default value when Ollama not installed
        ollama_api_url = "http://localhost:11434/api/generate"
        ollama_timeout = "180"
        ollama_max_length = "32000"

    # 6. Get language preference for Whisper
    print("\n--- Language Settings ---")
    print("Common languages: en (English), es (Spanish), fr (French), de (German), it (Italian), pt (Portuguese)")
    print("For full list: https://github.com/openai/whisper#available-models-and-languages")
    language_choice = input("Enter language code for Whisper (or 'auto' for detection) [default: auto]: ").lower() or "auto"

    # Get auto-action preferences
    print("\n--- Auto Actions ---")
    auto_copy = input("Auto copy to clipboard? (y/n) [default: n]: ").lower() or "n"
    auto_open = input("Auto open file after saving? (y/n) [default: n]: ").lower() or "n"

    if ollama_installed:
        auto_metadata = input("Auto generate AI summary/tags? (y/n) [default: n]: ").lower() or "n"
    else:
        auto_metadata = "n"  # Default to no when Ollama not installed

    # Get performance settings
    if is_basic_mode:
        print("\n--- Performance Settings ---")
        print("Using sensible defaults for performance settings:")
        print("  • Streaming buffer: 5 minutes (rolling buffer size)")
        print("  • Segment sizes: 30-60-90 seconds (min-target-max)")
        streaming_buffer = "300"
        streaming_min = "30"
        streaming_target = "60"
        streaming_max = "90"
    else:
        print("\n--- Performance Settings ---")
        print("\n1. Streaming Buffer Size")
        print("How much audio to keep in memory for context:")
        print("  • 300s (5m)   - Short sessions, low memory")
        print("  • 600s (10m)  - Balanced (recommended)")
        print("  • 900s (15m)  - Long sessions, high quality")
        streaming_buffer = input("Streaming buffer in seconds (60-1200) [default: 300]: ") or "300"

        print("\n2. Streaming Segment Durations")
        print("Control how audio is broken into chunks:")
        print("  • Min: Don't transcribe until at least this much speech")
        print("  • Target: Look for natural pauses around this duration")
        print("  • Max: Force break at this point (prevents memory issues)")
        streaming_min = input("Minimum segment duration (10-60s) [default: 30]: ") or "30"
        streaming_target = input("Target segment duration (30-120s) [default: 60]: ") or "60"
        streaming_max = input("Maximum segment duration (60-180s) [default: 90]: ") or "90"
    
    # Get microphone device selection
    print("\n--- Microphone Device Selection ---")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = []
        print("\nAvailable audio input devices:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']}")
                input_devices.append(i)
        
        if input_devices:
            device_choice = input("Enter device number (-1 for system default) [default: -1]: ").strip() or "-1"
            try:
                device_num = int(device_choice)
                if device_num == -1 or device_num in input_devices:
                    mic_device = str(device_num)
                else:
                    print("⚠️ Invalid device number, using system default")
                    mic_device = "-1"
            except ValueError:
                print("⚠️ Invalid input, using system default")
                mic_device = "-1"
        else:
            print("⚠️ No audio input devices found, using system default")
            mic_device = "-1"
    except ImportError:
        print("⚠️ sounddevice not available, using system default")
        mic_device = "-1"

    # Strip all input values to prevent spacing issues
    save_path = save_path.strip()
    output_format = output_format.strip()
    model_choice = model_choice.strip()
    language_choice = language_choice.strip()
    ollama_model = ollama_model.strip()
    rec_command = rec_command.strip()
    mic_device = str(mic_device).strip()
    streaming_buffer = str(streaming_buffer).strip()
    streaming_min = str(streaming_min).strip()
    streaming_target = str(streaming_target).strip()
    streaming_max = str(streaming_max).strip()
    ollama_api_url = ollama_api_url.strip()
    ollama_timeout = str(ollama_timeout).strip()
    ollama_max_length = str(ollama_max_length).strip()
    
    # Write configuration to .env file
    with open(".env", "w") as f:
        f.write(f"SAVE_PATH='{save_path}'\n")
        f.write(f"OUTPUT_FORMAT='{output_format}'\n")
        f.write(f"WHISPER_MODEL='{model_choice}'\n")
        f.write(f"WHISPER_LANGUAGE='{language_choice}'\n")
        
        # Write Ollama settings with appropriate comments
        if ollama_installed:
            f.write(f"OLLAMA_MODEL='{ollama_model}'\n")
            f.write(f"AUTO_METADATA={'true' if auto_metadata == 'y' else 'false'}\n")
        else:
            f.write(f"# Requires Ollama - install from https://ollama.ai\n")
            f.write(f"OLLAMA_MODEL='{ollama_model}'\n")
            f.write(f"# Requires Ollama - install from https://ollama.ai\n")
            f.write(f"AUTO_METADATA=false\n")
        
        f.write(f"COMMAND_NAME='{rec_command}'\n")
        f.write(f"AUTO_COPY={'true' if auto_copy == 'y' else 'false'}\n")
        f.write(f"AUTO_OPEN={'true' if auto_open == 'y' else 'false'}\n")

        # Obsidian Integration
        f.write(f"# Obsidian Integration\n")
        f.write(f"OBSIDIAN_ENABLED={'true' if obsidian_enabled else 'false'}\n")
        f.write(f"OBSIDIAN_VAULT_PATH='{obsidian_vault_path}'\n")

        f.write(f"DEFAULT_MIC_DEVICE={mic_device}\n")

        # Streaming transcription settings (now the default mode)
        f.write(f"# Streaming transcription settings\n")
        f.write(f"STREAMING_BUFFER_SIZE_SECONDS={streaming_buffer}\n")
        f.write(f"STREAMING_MIN_SEGMENT_DURATION={streaming_min}\n")
        f.write(f"STREAMING_TARGET_SEGMENT_DURATION={streaming_target}\n")
        f.write(f"STREAMING_MAX_SEGMENT_DURATION={streaming_max}\n")

        # Ollama AI settings
        f.write(f"# Ollama AI settings\n")
        f.write(f"OLLAMA_API_URL='{ollama_api_url}'\n")
        f.write(f"OLLAMA_TIMEOUT={ollama_timeout}\n")
        f.write(f"OLLAMA_MAX_CONTENT_LENGTH={ollama_max_length}\n")

    print("\n🎉 Setup complete! Configuration saved to .env")

    # Display configuration summary
    display_config_summary()

    print("The necessary aliases have been prepared.")
    print("Please restart your terminal or run 'source ~/.zshrc' to use them.")

if __name__ == "__main__":
    main()