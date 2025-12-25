# Rejoice Slim 🎙️
### Record. Jot. Voice.

**Free long-form transcription that runs offline - no limits, no data loss.**

> ⚠️ **Currently macOS only** - See [OS_AGNOSTIC_ROADMAP.md](Docs/Future_Thoughts/OS_AGNOSTIC_ROADMAP.md) for Linux/Windows support plans

Rejoice Slim is a voice-to-text tool that runs entirely on your Mac. Perfect for **Obsidian users** who want to capture thoughts, meetings, and ideas as voice notes that automatically become searchable Markdown files in their vault.

## ✨ What It Does

- 🎤 **One-command recording** - Start transcribing with `rec`
- 🆔 **Smart ID system** - Easy-to-reference transcripts with 6-digit IDs
- ➕ **Append to transcripts** - Add to existing recordings with `rec -000001`
- 🤖 **AI-powered analysis** - Generate filenames, summaries, and tags
- 📝 **Obsidian integration** - Smart vault selection and URI-based opening
- 🎙️ **Streaming transcription** - Processes audio continuously while recording
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
rec                              # Start recording (streaming, real-time transcription)
rec -000001                      # Append to existing transcript by ID
rec -l, --list                   # Show all transcripts with their IDs
rec -v 000001, --view 000001     # View content of transcript by ID
rec -g 000001, --genai 000001    # AI analysis: extract themes, questions, actions
rec -o, --open-folder            # Open transcripts folder
rec -s, --settings               # Configure settings
rec --audio 000001               # Show audio files for transcript ID
rec --reprocess 000001           # Reprocess transcript from audio
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

- **[🔧 Installation Guide](Docs/INSTALLATION.md)** - Detailed setup options and troubleshooting (macOS)
- **[📖 How to Use](Docs/USAGE.md)** - Complete user guide with examples
- **[⚙️ Settings](Docs/SETTINGS.md)** - Configuration options and customization
- **[📦 Dependencies](Docs/DEPENDENCIES.md)** - Package details and security information
- **[🏗️ Architecture](Docs/ARCHITECTURE.md)** - System design and developer guide
- **[🗺️ OS-Agnostic Roadmap](Docs/Future_Thoughts/OS_AGNOSTIC_ROADMAP.md)** - Future cross-platform support plans

---

## 🚀 Ready to Start?

1. **Install** following the [Installation Guide](Docs/INSTALLATION.md)
2. **Learn** the basics in [How to Use](Docs/USAGE.md)
3. **Customize** with the [Settings Guide](Docs/SETTINGS.md)

**Questions?** Check the documentation links above or create an issue on GitHub.