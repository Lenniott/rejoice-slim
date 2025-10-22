# Rejoice - Local Voice Transcriber 🎙️

**Turn your voice into searchable notes - completely offline and private.**

Rejoice is a voice-to-text tool that runs entirely on your computer. Perfect for **Obsidian users** who want to capture thoughts, meetings, and ideas as voice notes that automatically become searchable Markdown files in their vault.

## ✨ What It Does

- 🎤 **One-command recording** - Start transcribing with `rec`
- 🆔 **Smart ID system** - Easy-to-reference transcripts with descriptive names
- ➕ **Append to transcripts** - Add to existing recordings with `rec -000001`
- 🤖 **AI-powered analysis** - Hierarchical summarization extracts key themes, questions, and actions
- 📝 **Obsidian-ready** - Markdown format with YAML frontmatter
- 🔄 **Real-time transcription** - See your words appear as you speak
- 🎯 **Smart auto-stop** - Automatically stops when no speech detected
- ⚡ **Short commands** - Use `-l`, `-v`, `-g`, `-s` for quick access
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
rec -l                           # Show all transcripts with their IDs
rec -v 000001                    # View content of transcript by ID
rec -g 000001                    # AI analysis: extract themes, questions, actions
rec -s                           # Configure settings
```

### AI-Powered Features
```bash
rec -g /path/to/file.md         # Analyze any text file with AI
rec -g 000042                   # Analyze transcript by ID
# Extracts: main themes, key questions, action items, narrative threads
# Uses hierarchical processing for large content (30k+ characters)
# Generates intelligent filenames and tags automatically
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