# Rejoice - Local Voice Transcriber 🎙️

A simple, privacy-focused voice transcription tool that runs entirely on your local machine. Record audio, get AI-powered transcriptions, and automatically organize your files - all without sending your data to the cloud.

## ✨ Features

- 🎤 **One-command recording** - Start transcribing with a simple command
- 🤖 **AI-powered naming** - Automatically generates descriptive filenames using Ollama
- 📝 **Multiple formats** - Save as Markdown or plain text
- 🔧 **Easy configuration** - Interactive settings menu to customize everything
- 🏠 **100% local** - Your voice data never leaves your computer
- 🚀 **Zero setup friction** - One script installs and configures everything

## 🚀 Quick Start

### Prerequisites
You need these three things installed first:

1. **Python 3** - Download from [python.org/downloads](https://www.python.org/downloads/)
2. **Ollama** - Download from [ollama.com/download](https://ollama.com/download)
   - After installing, test it by running: `ollama run llama3`
3. **Homebrew** (macOS only) - Install from [brew.sh](https://brew.sh)

### Installation

1. **Download this project** and navigate to the folder in your terminal
2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
3. **Follow the interactive prompts** (or just press Enter for sensible defaults)
4. **Restart your terminal** or run `source ~/.zshrc`

That's it! 🎉

## 📋 Usage

After setup, you'll have three new commands available:

### `rec` - Start Recording
```bash
rec
```
- Press Enter when you want to stop recording
- Automatic transcription with Whisper
- AI-generated filename using Ollama
- Option to copy text or open the file

### `rec-settings` - Change Configuration  
```bash
rec-settings
```
Interactive menu to change:
- Whisper model (tiny/base/small/medium/large)
- Ollama model (llama3, qwen3:0.6b, phi3, etc.)
- Output format (Markdown or plain text)
- Save location

### `open-transcripts` - View Your Files
```bash
open-transcripts
```
Opens your transcripts folder in Finder/Explorer.

## 🔧 Configuration

The setup creates a `.env` file with your preferences:

```bash
SAVE_PATH='/path/to/your/transcripts'
OUTPUT_FORMAT='md'
WHISPER_MODEL='small'
OLLAMA_MODEL='llama3'
COMMAND_NAME='rec'
```

You can change these anytime using `rec-settings` or by editing the file directly.

## 🎯 Model Recommendations

**Whisper Models:**
- `tiny` - Fastest, least accurate
- `base` - Good balance for most uses  
- `small` - **Recommended** - Great accuracy/speed balance
- `medium` - Higher accuracy, slower
- `large` - Best accuracy, slowest

**Ollama Models:**
- `llama3` - **Recommended** - Great for filename generation
- `qwen3:0.6b` - Very fast, good for quick testing
- `phi3` - Good alternative to llama3
- `gemma` - Another solid option

## 🛠️ Troubleshooting

**Command not found?**
- Make sure you restarted your terminal after setup
- Or run: `source ~/.zshrc`

**Ollama connection issues?**
- Check Ollama is running: `ollama list`
- Try pulling a model: `ollama pull llama3`

**Audio recording problems?**
- Check microphone permissions in System Preferences
- Try a different audio input device

**Want to reinstall?**
- Just run `./setup.sh` again - it cleans up old configurations automatically

## 🔒 Privacy

Everything runs locally on your machine:
- ✅ Audio files processed locally with Whisper
- ✅ Filename generation uses local Ollama models
- ✅ No data sent to external services
- ✅ You control where files are saved

## 📁 File Structure

```
rejoice/
├── setup.sh           # One-command installation
├── configure.py       # Interactive configuration
├── requirements.txt   # Python dependencies
├── src/
│   └── transcribe.py  # Main transcription logic
└── .env              # Your settings (created during setup)
```