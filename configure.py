# configure.py

import os
import shutil

def main():
    print("--- 🎙️ Welcome to the Local Transcriber Setup ---")
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
    rec_command = input("Enter the command you'll use to start recording [default: rec]: ") or "rec"

    # 2. Get the directory to save transcripts
    default_path = os.path.join(os.path.expanduser("~"), "Documents", "transcripts")
    save_path = input(f"Enter the full path to save transcripts [default: {default_path}]: ") or default_path
    os.makedirs(save_path, exist_ok=True)
    print(f"✅ Transcripts will be saved in: {save_path}")

    # 3. Get the default output format
    output_format = ""
    while output_format not in ["md", "txt"]:
        output_format = input("Choose the default output format (md/txt) [default: md]: ") or "md"

    # 4. Choose Whisper model
    models = ["tiny", "base", "small", "medium", "large"]
    model_choice = ""
    while model_choice not in models:
        model_choice = input(f"Choose a Whisper model {models} [default: small]: ") or "small"
    print("\nDownloading the Whisper model now. This might take a moment...")
    # This part triggers the download during setup
    import whisper
    whisper.load_model(model_choice)
    print("✅ Model downloaded successfully.")

    # 5. Get Ollama model for naming (only if Ollama is installed)
    if ollama_installed:
        ollama_model = input("Enter the Ollama model to use for file naming (e.g., gemma3:4b) [default: gemma3:4b]: ") or "gemma3:4b"
    else:
        ollama_model = "gemma3:4b"  # Default value when Ollama not installed

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

    # Get real-time chunking settings
    if is_basic_mode:
        print("\n--- Performance Settings ---")
        print("Using sensible defaults for performance settings:")
        print("  • Chunk duration: 10 seconds (how often you see updates)")
        print("  • No speech detection: 2 minutes (auto-stop duration)")
        print("  • Advanced settings: Optimized automatically")
        chunk_duration = "10"
        silence_duration = "120"
    else:
        print("\n--- Performance Settings ---")
        print("Chunk Duration: How often you see transcription updates")
        print("  • Shorter (5-8s): More frequent updates")
        print("  • Default (10s): Smooth, balanced experience")  
        print("  • Longer (15-30s): Less frequent updates")
        chunk_duration = input("Chunk duration in seconds (5-30) [default: 10]: ") or "10"
        
        print("\nNo Speech Detection: Automatically stop recording when no speech detected")
        print("  • Default (120s): 2 minutes of no speech")
        print("  • Shorter (60s): 1 minute - stops sooner")
        print("  • Longer (180s): 3 minutes - waits longer")
        silence_duration = input("No speech duration in seconds (30-300) [default: 120]: ") or "120"
    
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
    chunk_duration = str(chunk_duration).strip()
    silence_duration = str(silence_duration).strip()
    mic_device = str(mic_device).strip()
    
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
        f.write(f"CHUNK_DURATION_SECONDS={chunk_duration}\n")
        f.write(f"SILENCE_DURATION_SECONDS={silence_duration}\n")
        f.write(f"DEFAULT_MIC_DEVICE={mic_device}\n")
        
        # Hardcoded advanced settings (optimized defaults)
        f.write(f"# Advanced settings - optimized for best performance\n")
        f.write(f"CHUNK_OVERLAP_SECONDS=2.5\n")
        f.write(f"TRANSCRIPTION_WORKER_THREADS=2\n")
        f.write(f"MAX_RETRY_ATTEMPTS=4\n")

    print("\n🎉 Setup complete! Configuration saved to .env")
    print("The necessary aliases have been prepared.")
    print("Please restart your terminal or run 'source ~/.zshrc' to use them.")

if __name__ == "__main__":
    main()