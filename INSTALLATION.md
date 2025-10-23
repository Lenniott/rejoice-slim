# 🔧 Installation Guide

**← [Back to Home](README.md)**

## 🚀 Quick Start

### One-Command Installation (Recommended)
```bash
curl -fsSL https://raw.githubusercontent.com/benjamayden/rejoice-slim/main/setup.sh | bash
```
This downloads, installs dependencies, and sets up everything automatically.

### Alternative Installation Options

**Option 1: Manual Download**
```bash
# Clone the repository
git clone https://github.com/benjamayden/rejoice-slim.git
cd rejoice-slim

# Run setup script
chmod +x setup.sh
./setup.sh
```

**Option 2: Audio-Only Mode (No Homebrew)**
```bash
# If you prefer not to install Homebrew dependencies
git clone https://github.com/benjamayden/rejoice-slim.git
cd rejoice-slim
pip install -r requirements.txt
python src/transcribe.py --config  # Configure output directory
```

## 📋 Prerequisites

You need these installed first:

### 1. Python 3
**Download:** [python.org/downloads](https://www.python.org/downloads/)
- **Required version:** Python 3.8 or newer
- **Check if installed:** `python3 --version`

### 2. Homebrew (macOS only)
**Install:** [brew.sh](https://brew.sh)
- **Why Homebrew?** Enables reliable microphone access while other apps (Zoom, Spotify) are running
- **What it installs:** PortAudio library for professional audio recording
- **Security:** Downloads precompiled, signed packages; only installs audio I/O library
- **Skip if needed:** Basic transcription works without it (may have audio compatibility issues)

### 3. Ollama (Optional)
**Download:** [ollama.com/download](https://ollama.com/download)
- **Required for:** Advanced AI features (intelligent analysis, hierarchical summarization, theme extraction)
- **Test installation:** `ollama run gemma3:4b`
- **Without Ollama:** Basic transcription works perfectly, just with timestamp-based filenames

## ⚙️ Installation Modes

The setup offers two installation modes:

### Basic Mode (Recommended)
- **Quick setup** with sensible defaults
- **No advanced configuration** - just works out of the box
- **Perfect for new users** who want to start transcribing immediately
- **Uses optimized defaults** for chunking and performance settings
- **Skips Ollama questions** if not installed

### Detailed Mode
- **Full configuration** of all advanced settings
- **Chunking parameters** - customize duration, overlap, worker threads
- **Voice Activity Detection** - configure silence detection
- **AI settings** - configure Ollama models and auto-metadata (if installed)
- **Perfect for power users** who want fine-grained control

> 💡 **Tip**: You can always change settings later using `rec-settings` command

## 🔧 What the Setup Does

### For Full Audio Recording (Recommended):
1. ✅ **Installs Python dependencies** - Whisper, sounddevice, and audio tools
2. ✅ **Installs Homebrew audio stack** - PortAudio for reliable microphone access
3. ✅ **Downloads Whisper models** - AI transcription models (1-3 GB)
4. ✅ **Sets up Ollama** - Local AI for smart naming (optional)
5. ✅ **Creates shell alias** - `rec` command in your terminal
6. ✅ **Configures output directory** - Choose where transcriptions are saved

### For Audio-Only Mode:
- ⚡ **Skip Homebrew** - Uses basic audio recording (may have compatibility issues)
- 📦 **Python packages only** - Faster but less reliable audio
- 🎯 **Core functionality** - Still provides full transcription

## ✅ Post-Installation

After setup completes:

```bash
# Start transcribing (creates 000001.md, 000002.md, etc.)
rec

# List all transcripts with their IDs
rec --list

# View or append to existing transcript
rec --show 000001
rec -000001

# Configure settings
python src/transcribe.py --settings
```

The setup script creates an alias for easy access. If it doesn't work, restart your terminal or run:
```bash
source ~/.zshrc  # or ~/.bashrc
```

## 🔍 Troubleshooting

### Common Issues

**"Command not found: rec"**
- Restart your terminal
- Or run: `source ~/.zshrc` (or `~/.bashrc`)

**"No module named 'sounddevice'"**
- Run: `pip install -r requirements.txt`
- Make sure you're using the right Python environment

**"PortAudio not found"**
- Install Homebrew: [brew.sh](https://brew.sh)
- Run: `brew install portaudio`

**"Ollama connection failed"**
- Install Ollama: [ollama.com/download](https://ollama.com/download)
- Test: `ollama run gemma3:4b`
- Transcription still works without Ollama

### Getting Help

1. **Check the logs** - Error messages usually point to the issue
2. **Try audio-only mode** - Skip Homebrew if having installation issues  
3. **Test components separately** - Try Python, Whisper, and Ollama independently
4. **Create an issue** - Include your OS, Python version, and error messages

---

**← [Back to Home](README.md)** | **Next: [How to Use →](USAGE.md)**