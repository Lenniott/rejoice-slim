# Rejoice - Local Voice Transcriber 🎙️

**Turn your voice into searchable notes - completely offline and private.**

Rejoice is a voice-to-text tool that runs entirely on your computer. Perfect for **Obsidian users** who want to capture thoughts, meetings, and ideas as voice notes that automatically become searchable Markdown files in their vault.

## 🔒 Your Data Stays Yours

**Verified Privacy Guarantees:**
- ✅ **All audio processing on your device** - Whisper runs locally
- ✅ **No OpenAI API calls** - Uses local Whisper models
- ✅ **No cloud AI services** - Ollama runs locally (optional)
- ✅ **No internet required** for transcription
- ✅ **Zero telemetry or analytics**
- ✅ **Zero external API calls**
- ✅ **Files saved only where you choose**
- ✅ **Open source** - verify the code yourself

**Technical Verification:**
The only network request is to `localhost:11434` (local Ollama). See `src/transcribe.py` line 38.

## ✨ What It Does

- 🎤 **One-command recording** - Start transcribing with a simple command
- 🤖 **AI-powered naming** - Automatically generates descriptive filenames using Ollama (optional)
- 📝 **Obsidian-optimized** - Markdown format designed for seamless Obsidian integration
- 🔄 **Real-time transcription** - See your words appear as you speak
- 🎯 **Smart auto-stop** - Automatically stops when no speech detected
- 🔧 **Easy configuration** - Interactive settings menu to customize everything
- 🏠 **100% local** - Your voice data never leaves your computer
- 🚀 **Zero setup friction** - One script installs and configures everything

## 🎯 What It's Good For

**Perfect for capturing:**
- 📝 **Meeting notes** - Record and transcribe important discussions
- 🎙️ **Voice journaling** - Daily thoughts and reflections
- 💡 **Quick idea capture** - Brainstorming and creative thinking
- 📚 **Lecture transcription** - Educational content and presentations
- 🔍 **Interview transcription** - Research and documentation
- 📖 **Obsidian workflow** - Seamless integration with your knowledge vault

**Why Rejoice?**
- **Privacy-first** - Your sensitive conversations stay on your device
- **Offline capable** - Works without internet connection
- **Obsidian-ready** - Markdown files integrate perfectly
- **Real-time feedback** - See transcription as you speak
- **Smart automation** - AI-generated filenames and summaries

## 🚀 Quick Start

### Prerequisites
You need these installed first:

1. **Python 3** - Download from [python.org/downloads](https://www.python.org/downloads/)
2. **Homebrew** (macOS only) - Install from [brew.sh](https://brew.sh)
3. **Ollama** (optional) - Download from [ollama.com/download](https://ollama.com/download)
   - Required only for AI features (smart filenames, summaries, tags)
   - After installing, test it by running: `ollama run gemma3:4b`
   - **Without Ollama**: Basic transcription works perfectly, just with timestamp-based filenames

### Installation

1. **Download this project** and navigate to the folder in your terminal
2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
3. **Follow the interactive prompts** (or just press Enter for sensible defaults)
4. **Restart your terminal** or run `source ~/.zshrc`

That's it! 🎉

## ⚙️ Installation Modes

The setup offers two installation modes to suit different user needs:

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

## ⚙️ Setup Settings Explained

### Basic Mode Settings
When you choose **Basic mode**, these settings are configured automatically with sensible defaults:

**🎤 Recording Command** (`rec`)
- The command you'll type to start recording
- Default: `rec` (you can change this to anything you prefer)

**📁 Save Path** 
- Where your transcript files will be saved
- Default: `~/Documents/transcripts`
- **Pro tip**: Set this to your Obsidian vault folder for seamless integration

**📝 Output Format** (`md` or `txt`)
- File format for your transcripts
- Default: `md` (Markdown) - **Recommended for Obsidian users**
- Alternative: `txt` (plain text)

**🤖 Whisper Model** (`tiny`, `base`, `small`, `medium`, `large`)
- AI model used for transcription accuracy
- Default: `small` - **Best balance of speed and accuracy**
- Larger models = more accurate but slower

**🌍 Language** (`auto` or language code)
- Language for transcription
- Default: `auto` (automatic detection)
- Examples: `en` (English), `es` (Spanish), `fr` (French)

**🎯 Auto Actions** (`y` or `n`)
- **Auto Copy**: Copy transcript to clipboard after saving
- **Auto Open**: Open the saved file after transcription
- **Auto Metadata**: Generate AI summaries and tags (requires Ollama)

**⚡ Performance Settings** (configured automatically)
- **Chunk Duration**: 10 seconds - How often you see transcription updates
- **No Speech Detection**: 2 minutes - Auto-stop when no speech detected
- **Advanced settings**: Optimized automatically for best performance

### Detailed Mode Settings
When you choose **Detailed mode**, you can configure all the above settings plus:

**🎤 Microphone Device**
- Select specific audio input device
- Default: System default
- Useful if you have multiple microphones

**🤖 Ollama Model** (if Ollama installed)
- AI model for smart filenames and metadata
- Default: `gemma3:4b` - **Recommended for best results**
- Alternatives: `llama3`, `qwen3:0.6b`, `phi3`

**⚡ Performance Settings** (fully customizable)
- **Chunk Duration** (5-30 seconds): How often you see transcription updates
  - Shorter = more frequent updates, longer = less frequent updates
- **No Speech Detection** (30-300 seconds): Auto-stop when no speech detected
  - Shorter = stops sooner, longer = waits longer before stopping

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

## ⚙️ How It Works

**For Technical Users:**

Rejoice uses a sophisticated real-time audio processing pipeline:

1. **Audio Chunking** - Records audio in 10-second chunks with 2.5-second overlap
2. **Voice Activity Detection** - Uses RMS energy and signal variance to detect speech
3. **Parallel Transcription** - 2 worker threads process chunks with Whisper
4. **Real-time Assembly** - Chunks are transcribed and assembled in real-time
5. **Smart Auto-stop** - Stops recording after 2 minutes of no speech detected
6. **AI Enhancement** - Optional Ollama integration for smart filenames and metadata

**Technical Architecture:**
- **Whisper**: Local speech-to-text (no API calls)
- **Ollama**: Local AI for metadata generation (optional)
- **VAD Service**: Energy-based voice activity detection
- **Worker Pool**: Parallel transcription processing
- **Audio Chunker**: Real-time audio segmentation

**Performance Optimizations:**
- 2.5-second chunk overlap ensures no word loss
- 2 worker threads optimal for most systems
- 4 retry attempts for failed chunks
- Configurable chunk duration (5-30 seconds)
- Smart silence detection (30-300 seconds)

## 🔗 Obsidian Integration

Rejoice is designed with **Obsidian users** in mind. The Markdown output format is optimized for seamless integration with your Obsidian vault.

### Setting Up for Obsidian

1. **During setup**, set your save path to your Obsidian vault:
   ```
   /path/to/your/obsidian/vault/transcripts
   ```

2. **AI-generated content** works perfectly with Obsidian:
   - **Smart filenames** become searchable note titles
   - **Auto-generated tags** appear as Obsidian tags
   - **Summaries** can be used for note previews
   - **Timestamps** are Obsidian-friendly format

3. **Workflow example**:
   ```
   Voice note → Rejoice → Markdown file → Obsidian vault → Searchable knowledge
   ```

### Obsidian-Specific Features

- **Metadata headers** in YAML format for better organization
- **Consistent timestamping** for chronological sorting
- **Clean formatting** that renders beautifully in Obsidian
- **Tag generation** that works with Obsidian's tag system

> 💡 **Pro tip**: Set up a daily note template in Obsidian that includes a link to your transcripts folder for easy access!

## 🔧 Configuration

The setup creates a `.env` file with your preferences. Here are all available settings:

### Basic Settings
```bash
SAVE_PATH='/path/to/your/transcripts'     # Where to save transcript files
OUTPUT_FORMAT='md'                        # File format: 'md' or 'txt'
WHISPER_MODEL='small'                     # Whisper model: tiny/base/small/medium/large
WHISPER_LANGUAGE='auto'                   # Language code or 'auto' for detection
COMMAND_NAME='rec'                        # Your recording command name
DEFAULT_MIC_DEVICE='-1'                   # Audio input device (-1 = system default)
```

### Auto Actions
```bash
AUTO_COPY=false                          # Auto copy transcript to clipboard
AUTO_OPEN=false                          # Auto open file after saving
AUTO_METADATA=false                      # Auto generate AI summary/tags (requires Ollama)
```

### AI Settings (requires Ollama)
```bash
OLLAMA_MODEL='gemma3:4b'                 # Ollama model for AI features
# Note: These settings are disabled if Ollama is not installed
```

### Performance Settings
```bash
CHUNK_DURATION_SECONDS=10                # How often you see transcription updates
SILENCE_DURATION_SECONDS=120             # No speech detection duration (2 minutes)

# Advanced settings (optimized defaults - power users can override)
CHUNK_OVERLAP_SECONDS=2.5                # Ensures no word loss at boundaries
TRANSCRIPTION_WORKER_THREADS=2           # Optimal for most systems
MAX_RETRY_ATTEMPTS=4                     # Handles temporary failures
```

### Settings Management

All settings can be changed using the `rec-settings` command, which provides a **categorized menu system**:

**📝 Transcription Settings**
- Whisper model and language selection

**📁 Output Settings** 
- File format, save path, auto-copy, auto-open, auto-metadata

**🤖 AI Settings**
- Ollama model selection and auto-metadata (if Ollama installed)

**🎤 Audio Settings**
- Microphone device selection

**⚡ Performance Settings**
- Chunk duration (how often updates appear)
- No speech detection duration (auto-stop timing)

> 💡 **Pro tip**: The settings menu is organized into logical categories to make it easy to find what you're looking for. Advanced settings include helpful descriptions and warnings about performance implications.

## 🎯 Model Recommendations

**Whisper Models:**
- `tiny` - Fastest, least accurate
- `base` - Good balance for most uses  
- `small` - **Recommended** - Great accuracy/speed balance
- `medium` - Higher accuracy, slower
- `large` - Best accuracy, slowest

**Ollama Models:**
- `gemma3:4b` - **Recommended** - Great for filename generation and metadata
- `llama3` - Good alternative with strong performance
- `qwen3:0.6b` - Very fast, good for quick testing
- `phi3` - Another solid option

## 🔧 Advanced Configuration

**For Power Users:**

Advanced settings are hardcoded with optimal defaults but can be overridden by editing the `.env` file directly:

**Why These Defaults?**
- **2.5-second overlap**: Ensures no word is split across chunks (longest English pronunciation)
- **2 worker threads**: Sweet spot for real-time processing (more = diminishing returns)
- **4 retry attempts**: Handles temporary failures without getting stuck
- **2-minute no speech detection**: Reasonable wait time before auto-stop

**Override Instructions:**
1. Edit the `.env` file in your project directory
2. Modify the advanced settings as needed
3. Restart the application to apply changes

> 📖 **Technical details**: See `CHUNKING_ARCHITECTURE.md` for in-depth technical documentation

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
- ✅ **Audio files processed locally** with Whisper
- ✅ **AI features use local Ollama models** (when installed)
- ✅ **Real-time chunking** happens entirely on your device
- ✅ **No data sent to external services** - ever
- ✅ **You control where files are saved** - including directly to Obsidian vaults
- ✅ **Voice Activity Detection** runs locally without cloud processing

## 📁 File Structure

```
rejoice/
├── setup.sh                    # One-command installation
├── configure.py                # Interactive configuration with modes
├── requirements.txt            # Python dependencies
├── prompts.json               # AI prompt templates
├── templates.json             # File templates
├── CHUNKING_ARCHITECTURE.md   # Technical documentation
├── src/
│   ├── transcribe.py          # Main transcription logic
│   ├── audio_chunker.py       # Real-time chunking system
│   ├── transcription_worker.py # Parallel transcription workers
│   └── vad_service.py         # Voice Activity Detection
└── .env                       # Your settings (created during setup)
```