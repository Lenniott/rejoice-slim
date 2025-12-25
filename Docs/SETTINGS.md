# ⚙️ Rejoice Slim Settings Guide

**← [Back to Home](README.md)**

> ⚠️ **macOS Only** - These settings are for macOS. See [OS_AGNOSTIC_ROADMAP.md](Docs/Future_Thoughts/STANDALONE_APP_ANALYSIS.md) for other platforms.

## 🎛️ Configuration Menu

Access settings anytime with:
```bash
rec --settings
# or
rec -s
```

This opens an interactive menu organized into 3 intuitive categories:
- **🎯 Core** - Commonly adjusted settings (path, format, models, mic)
- **🎨 Casual** - Quality of life features (auto-actions, command name)
- **⚙️ Advanced** - Technical settings (streaming, Ollama config)

## 🎯 Core Settings (Commonly Adjusted)

These are the settings you'll adjust most often. Access them via: `rec -s` → Option 1

### 📁 Save Path
- **What it is:** Where your transcript files are saved
- **Default:** `~/Documents/transcripts`
- **Pro tip:** Set to your Obsidian vault for seamless integration
- **Examples:** `~/MyVault/Voice Notes`, `~/Desktop/Transcripts`
- **How to change:** Settings → Core → Change Save Path

### 📝 Output Format
- **Options:** `md` (Markdown) or `txt` (plain text)
- **Default:** `md` - **Recommended for Obsidian users**
- **Markdown benefits:** YAML frontmatter, tags, better organization
- **Plain text:** Simple format, works everywhere
- **How to change:** Settings → Core → Change Output Format

### 🎙️ Whisper Model
Control transcription accuracy and speed:

| Model  | Size | Speed    | Accuracy | Best For |
|--------|------|----------|----------|----------|
| `tiny` | 39MB | Fastest  | Good     | Quick notes, testing |
| `base` | 74MB | Fast     | Better   | General use, older computers |
| `small`| 244MB| Balanced | Very Good| **Recommended default** |
| `medium`| 769MB| Slower  | Excellent| Professional use |
| `large`| 1550MB| Slowest | Best     | Critical accuracy needs |

- **Default:** `small`
- **Recommendation:** Start with `small`, upgrade to `medium` if you need better accuracy
- **How to change:** Settings → Core → Change Whisper Model

### 🌍 Language
- **Default:** `auto` (automatic detection)
- **Manual options:** `en` (English), `es` (Spanish), `fr` (French), `de` (German), etc.
- **When to use manual:** Consistent accuracy for single-language environments
- **When to use auto:** Multi-language conversations or unsure
- **Supported:** 99+ languages
- **How to change:** Settings → Core → Change Language

### 🤖 Ollama Model
Controls AI-generated filenames and metadata:

| Model      | Size | Speed | Quality | Best For |
|------------|------|-------|---------|----------|
| `gemma3:4b`| 2.5GB| Fast  | Excellent| **Recommended default** |
| `llama3`   | 4.7GB| Medium| Excellent| Alternative to Gemma |
| `qwen3:0.6b`| 350MB| Fastest| Good   | Limited RAM/storage |
| `phi3`     | 2.3GB| Fast  | Very Good| Alternative option |

- **Default:** `gemma3:4b`
- **Without Ollama:** Files get timestamp names like `transcript_2024-10-20_14-30.md`
- **How to change:** Settings → Core → Change Ollama Model

### 🎤 Microphone Device
- **Default:** System Default (-1)
- **What it does:** Selects which audio input device to use
- **List devices:** Shows all available input devices with their numbers
- **How to change:** Settings → Core → Change Microphone Device

## 🎨 Casual Settings (Quality of Life)

These settings control convenience features. Access them via: `rec -s` → Option 2

### 🤖 Auto Metadata
- **Default:** Enabled (if Ollama installed)
- **What it does:** AI generates intelligent filename, summary, and tags automatically
- **Disable if:** Want manual control over metadata
- **Requirements:** Ollama must be installed and running
- **How to change:** Settings → Casual → Toggle Auto Metadata

### 🎤 Command Name
- **What it is:** The command you type to start recording
- **Default:** `rec`
- **Examples:** `record`, `transcribe`, `voice`, `tr`, `notes`
- **Why change it:** Avoid conflicts with other commands or personal preference
- **Safety features:** Won't overwrite existing commands, creates backup
- **How to change:** Settings → Casual → Change Command Name

### 📋 Auto Copy
- **Default:** Enabled
- **What it does:** Copies transcript text to clipboard after saving
- **Useful for:** Quickly pasting into other apps
- **Disable if:** Working with sensitive content
- **How to change:** Settings → Casual → Toggle Auto Copy

### 📂 Auto Open
- **Default:** Disabled
- **What it does:** Opens saved file in default editor after transcription
- **Useful for:** Immediate review and editing
- **Enable if:** You always edit transcripts after recording
- **How to change:** Settings → Casual → Toggle Auto Open

### 📝 Obsidian Integration
- **Default:** Not configured
- **What it does:** Seamlessly integrates with your Obsidian vault for opening transcripts
- **Setup workflow:**
  1. **Enable integration** - Answer "yes" when asked during setup
  2. **Vault selection** - System shows numbered list of directories from your save path
  3. **Pick your vault** - Select which directory is your Obsidian vault root
- **Example:**
  ```
  Save path: /Users/you/Documents/Obsidian/MyVault/Notes/Transcripts

  Detected directories from path:
  1. Users
  2. you
  3. Documents
  4. Obsidian
  5. MyVault  ← Select this
  6. Notes
  7. Transcripts

  Which directory is your Obsidian vault? [Enter number]: 5
  ```
- **Smart behavior:**
  - **When enabled:** Output format automatically set to `md` (Obsidian requires Markdown)
  - **Opens with URI:** Uses `obsidian://open?vault=...&file=...` for reliable vault integration
  - **Falls back gracefully:** If file is outside vault, opens with default app
- **How to configure:**
  - During setup: Automatically prompted after setting save path
  - In settings: `rec -s` → Casual → Configure Obsidian Integration
- **Reconfiguration:** Run settings anytime to change vault selection

## ⚙️ Advanced Settings (Technical)

These are technical performance and AI configuration settings. Access them via: `rec -s` → Option 3

### 🎚️ Streaming Buffer Size
- **Default:** 300 seconds (5 minutes)
- **What it does:** How much audio to keep in memory for context (rolling buffer)
- **Shorter (5m):** Short sessions, low memory
- **Balanced (10m):** Recommended for most use
- **Longer (15m):** Long sessions, high quality
- **Range:** 60-1200 seconds (1-20 minutes)
- **How to change:** Settings → Advanced → Configure Streaming Buffer Size

### 📊 Streaming Segments
Control how audio is broken into chunks for transcription:

- **Minimum (30s):** Don't transcribe until at least this much speech
- **Target (60s):** Look for natural pauses around this duration
- **Maximum (90s):** Force break at this point (prevents memory issues)

**Ranges:**
- Min: 10-60 seconds
- Target: 30-120 seconds
- Max: 60-180 seconds

**Rule:** min ≤ target ≤ max

**How to change:** Settings → Advanced → Configure Streaming Segment Durations

### 🔗 Ollama API URL
- **Default:** `http://localhost:11434/api/generate`
- **What it does:** URL endpoint for Ollama API requests
- **Change if:** Running Ollama on different port or remote server
- **How to change:** Settings → Advanced → Change Ollama API URL

### ⏱️ Ollama Timeout
- **Default:** 180 seconds (3 minutes)
- **What it does:** How long to wait for AI responses
- **Fast models (60s):** gemma3:270m, qwen3:0.6b
- **Medium models (180s):** gemma3:4b, llama3 (recommended)
- **Large models (300s):** llama3:70b
- **Range:** 30-600 seconds
- **How to change:** Settings → Advanced → Change Ollama Timeout

### 📏 Max AI Content Length
- **Default:** 32,000 characters
- **What it does:** Maximum content size for AI processing (truncates if larger)
- **Conservative (8,000):** For smaller models
- **Balanced (32,000):** Recommended default
- **Powerful (64,000):** For powerful setups
- **Range:** 1,000-200,000 characters
- **Note:** Content is truncated, not hierarchically summarized
- **How to change:** Settings → Advanced → Change Max AI Content Length

## 🎤 Audio Device Information

### Listing Available Devices
The settings menu shows available audio input devices when you select "Change Microphone Device":
```
Available audio input devices:
  0: Built-in Microphone
  1: USB Headset Microphone
  2: Blue Yeti USB Microphone
```

### Audio Quality
These are optimized automatically:
- **Sample Rate:** 16kHz (optimized for Whisper)
- **Channels:** Mono (speech recognition works best with single channel)
- **Bit Depth:** 16-bit (sufficient for voice)

> **Note:** Manual adjustment usually not needed - the system handles this automatically.

## 📄 Output Customization

### Markdown Format (Default)
```markdown
---
date: 2024-10-20 14:30
duration: 00:02:45
tags: [meeting-notes, project-planning, brainstorming]
summary: Discussion about new feature roadmap and timeline priorities
ai_model: gemma3:4b
whisper_model: small
---

# Project Planning Meeting - October 20th

[Transcription content here...]
```

### Plain Text Format
```
Date: 2024-10-20 14:30
Duration: 00:02:45
Tags: meeting-notes, project-planning, brainstorming
Summary: Discussion about new feature roadmap and timeline priorities

Project Planning Meeting - October 20th

[Transcription content here...]
```

## 📋 Settings Overview Display

When you run `rec --settings`, you'll see a comprehensive overview organized into the 3 categories:

```
======================================================================
📋 CURRENT CONFIGURATION OVERVIEW
======================================================================

🎯 CORE (Commonly Adjusted)
   Save Path:          /Users/you/Documents/transcripts
   Output Format:      md
   Whisper Model:      small
   Language:           en
   Ollama Model:       gemma3:4b
   Microphone:         System Default

🎨 CASUAL (Quality of Life)
   Auto Metadata:      ✅ Yes
   Command Name:       rec
   Auto Copy:          ✅ Yes
   Auto Open:          ✅ Yes
   Open in Obsidian:   ✅ Yes

⚙️ ADVANCED (Technical)
   Buffer Size:        5m 0s (rolling audio buffer)
   Segments:           30s-60s-90s (min-target-max chunks)
   Ollama API:         http://localhost:11434/api/generate
   Ollama Timeout:     3m 0s
   Max AI Content:     32,000 chars

======================================================================
💡 HOW TO CHANGE SETTINGS
======================================================================
   Run: rec --settings

   Then navigate to:
   • Option 1 - 🎯 Core settings (path, format, models, mic)
   • Option 2 - 🎨 Casual settings (auto-actions, command name)
   • Option 3 - ⚙️  Advanced settings (streaming, Ollama config)
======================================================================
```

## 🔧 Environment Variables

All settings are stored in `.env`. Advanced users can edit directly:

```bash
# Core Settings
SAVE_PATH='/Users/you/Documents/transcripts'
OUTPUT_FORMAT='md'
WHISPER_MODEL='small'
WHISPER_LANGUAGE='en'
OLLAMA_MODEL='gemma3:4b'
DEFAULT_MIC_DEVICE=-1

# Casual Settings
AUTO_METADATA=true
COMMAND_NAME='rec'
AUTO_COPY=true
AUTO_OPEN=true
OPEN_IN_OBSIDIAN=true

# Advanced Settings - Streaming
STREAMING_BUFFER_SIZE_SECONDS=300
STREAMING_MIN_SEGMENT_DURATION=30
STREAMING_TARGET_SEGMENT_DURATION=60
STREAMING_MAX_SEGMENT_DURATION=90

# Advanced Settings - Ollama
OLLAMA_API_URL='http://localhost:11434/api/generate'
OLLAMA_TIMEOUT=180
OLLAMA_MAX_CONTENT_LENGTH=32000
```

## 🔄 Backup and Reset

### Backup Settings
Your settings are stored in `.env`. Back it up:
```bash
cp .env .env.backup
```

### Reconfigure from Scratch
Re-run the setup wizard:
```bash
python configure.py
```

Choose either:
- **Basic mode:** Quick setup with sensible defaults
- **Detailed mode:** Configure all advanced settings

## 🛠️ Troubleshooting Settings

### "Settings not saving"
- Check file permissions in the project directory
- Ensure `.env` file is writable
- Try running `rec --settings` as your user (not sudo)

### "Command not found after changing alias"
- Reload shell: `source ~/.zshrc`
- Or restart your terminal
- Backup is automatically created at `~/.zshrc.backup`

### "AI features stopped working"
- Check Ollama is running: `ollama list`
- Test connection: `curl http://localhost:11434/api/version`
- Verify model exists: `ollama pull gemma3:4b`

### "Audio device not found"
- Go to Settings → Core → Change Microphone Device to see available devices
- Check system audio preferences
- Try system default (-1) if specific device doesn't work

### "Can't find settings menu option"
The new menu has 3 main categories:
- **Core (1):** Path, format, models, mic
- **Casual (2):** Auto-actions, command name
- **Advanced (3):** Streaming, Ollama config

If you're looking for a specific setting, check the category it logically belongs to.

---

**← [Back to Home](README.md)** | **Next: [Dependencies Guide →](DEPENDENCIES.md)**