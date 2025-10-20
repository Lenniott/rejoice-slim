# configure.py

import os

def main():
    print("--- 🎙️ Welcome to the Local Transcriber Setup ---")
    print("Let's configure your settings. Press Enter to accept defaults.\n")

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
        model_choice = input(f"Choose a Whisper model {models} [default: small]: ") or "base"
    print("\nDownloading the Whisper model now. This might take a moment...")
    # This part triggers the download during setup
    import whisper
    whisper.load_model(model_choice)
    print("✅ Model downloaded successfully.")

    # 5. Get Ollama model for naming
    ollama_model = input("Enter the Ollama model to use for file naming (e.g., gemma3:270m) [default: gemma3:270m]: ") or "gemma3:270m"

    # 6. Get language preference for Whisper
    print("\n--- Language Settings ---")
    print("Common languages: en (English), es (Spanish), fr (French), de (German), it (Italian), pt (Portuguese)")
    print("For full list: https://github.com/openai/whisper#available-models-and-languages")
    language_choice = input("Enter language code for Whisper (or 'auto' for detection) [default: auto]: ").lower() or "auto"

    # Get auto-action preferences
    print("\n--- Auto Actions ---")
    auto_copy = input("Auto copy to clipboard? (y/n) [default: n]: ").lower() or "n"
    auto_open = input("Auto open file after saving? (y/n) [default: n]: ").lower() or "n"
    auto_metadata = input("Auto generate AI summary/tags? (y/n) [default: n]: ").lower() or "n"

    # Write configuration to .env file
    with open(".env", "w") as f:
        f.write(f"SAVE_PATH='{save_path}'\n")
        f.write(f"OUTPUT_FORMAT='{output_format}'\n")
        f.write(f"WHISPER_MODEL='{model_choice}'\n")
        f.write(f"WHISPER_LANGUAGE='{language_choice}'\n")
        f.write(f"OLLAMA_MODEL='{ollama_model}'\n")
        f.write(f"COMMAND_NAME='{rec_command}'\n")
        f.write(f"AUTO_COPY={'true' if auto_copy == 'y' else 'false'}\n")
        f.write(f"AUTO_OPEN={'true' if auto_open == 'y' else 'false'}\n")
        f.write(f"AUTO_METADATA={'true' if auto_metadata == 'y' else 'false'}\n")

    print("\n🎉 Setup complete! Configuration saved to .env")
    print("The necessary aliases have been prepared.")
    print("Please restart your terminal or run 'source ~/.zshrc' to use them.")

if __name__ == "__main__":
    main()