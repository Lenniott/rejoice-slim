# Rejoice - Local Voice Transcriber 🎙️

**Turn your voice into searchable notes - completely offline and private.**

Rejoice is a voice-to-text tool that runs entirely on your computer. Perfect for **Obsidian users** who want to capture thoughts, meetings, and ideas as voice notes that automatically become searchable Markdown files in their vault.

## ✨ What It Does

- 🎤 **One-command recording** - Start transcribing with `rec`
- 🆔 **Smart ID system** - Easy-to-reference transcripts with descriptive names
- ➕ **Append to transcripts** - Add to existing recordings with `rec -000001`
- 📝 **Obsidian-ready** - Markdown format with YAML frontmatter
- 🔄 **Real-time transcription** - See your words appear as you speak
- 🎯 **Smart auto-stop** - Automatically stops when no speech detected
- 🏠 **100% local** - Your voice data never leaves your computer

## � Privacy First

- ✅ **All processing on your device** - Whisper + Ollama run locally
- ✅ **No cloud services** - Zero external API calls
- ✅ **Completely offline** - No internet required after setup
- ✅ **You control the data** - Files saved where you choose

## 🎯 Perfect For

- � **Meeting notes** and voice journaling
- � **Quick idea capture** and brainstorming  
- 📚 **Lecture transcription** and interviews
- 📖 **Obsidian workflow** integration

## 🚀 Quick Start

### Installation
```bash
curl -fsSL https://raw.githubusercontent.com/benjamayden/rejoice-slim/main/setup.sh | bash
```

### Basic Usage  
```bash
rec                              # Start recording (creates smart_filename_22102025_000001.md, etc.)
rec -000001                      # Append to existing transcript by ID
rec --list                       # Show all transcripts with their IDs
python src/transcribe.py --settings  # Configure settings
```

---

## 📚 Documentation

- **[🔧 Installation Guide](INSTALLATION.md)** - Detailed setup options and troubleshooting
- **[📖 How to Use](USAGE.md)** - Complete user guide with examples  
- **[⚙️ Settings](SETTINGS.md)** - Configuration options and customization
- **[📦 Dependencies](DEPENDENCIES.md)** - Package details and security information

---

## 🚀 Ready to Start?

1. **Install** following the [Installation Guide](INSTALLATION.md)
2. **Learn** the basics in [How to Use](USAGE.md)
3. **Customize** with the [Settings Guide](SETTINGS.md)

**Questions?** Check the documentation links above or create an issue on GitHub.