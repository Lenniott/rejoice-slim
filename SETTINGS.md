# ⚙️ Settings Guide

**← [Back to Home](README.md)**

## 🎛️ Configuration Menu

Access settings anytime with:
```bash
rec-settings
```

This opens an interactive menu to configure all aspects of Rejoice.

## 📋 Basic Settings

### 🎤 Recording Command
- **What it is:** The command you type to start recording
- **Default:** `rec`
- **Examples:** `record`, `transcribe`, `voice`, `tr`
- **Why change it:** Avoid conflicts with other commands

### 📁 Save Path
- **What it is:** Where your transcript files are saved
- **Default:** `~/Documents/transcripts`
- **Pro tip:** Set to your Obsidian vault for seamless integration
- **Examples:** `~/MyVault/Voice Notes`, `~/Desktop/Transcripts`

### 📝 Output Format
- **Options:** `md` (Markdown) or `txt` (plain text)
- **Default:** `md` - **Recommended for Obsidian users**
- **Markdown benefits:** YAML frontmatter, tags, better organization
- **Plain text:** Simple format, works everywhere

## 🤖 AI Models

### Whisper Model Selection
Control transcription accuracy and speed:

| Model  | Size | Speed    | Accuracy | Best For |
|--------|------|----------|----------|----------|
| `tiny` | 39MB | Fastest  | Good     | Quick notes, testing |
| `base` | 74MB | Fast     | Better   | General use, older computers |
| `small`| 244MB| Balanced | Very Good| **Recommended default** |
| `medium`| 769MB| Slower  | Excellent| Professional use |
| `large`| 1550MB| Slowest | Best     | Critical accuracy needs |

**Recommendation:** Start with `small`, upgrade to `medium` if you need better accuracy.

### Ollama Model Selection (Optional)
Controls AI-generated filenames and metadata:

| Model      | Size | Speed | Quality | Best For |
|------------|------|-------|---------|----------|
| `gemma3:4b`| 2.5GB| Fast  | Excellent| **Recommended default** |
| `llama3`   | 4.7GB| Medium| Excellent| Alternative to Gemma |
| `qwen3:0.6b`| 350MB| Fastest| Good   | Limited RAM/storage |
| `phi3`     | 2.3GB| Fast  | Very Good| Alternative option |

**Without Ollama:** Files get timestamp names like `transcript_2024-10-20_14-30.md`

## 🌍 Language Settings

### Language Detection
- **Default:** `auto` (automatic detection)
- **Manual options:** `en` (English), `es` (Spanish), `fr` (French), `de` (German), etc.
- **When to use manual:** Consistent accuracy for single-language environments
- **When to use auto:** Multi-language conversations or unsure

### Supported Languages
Whisper supports 99+ languages including:
- **Common:** English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean
- **Full list:** Available in the configuration menu

## 🎯 Auto Actions

### Auto Copy to Clipboard
- **Default:** Enabled
- **What it does:** Copies transcript text after saving
- **Useful for:** Quickly pasting into other apps
- **Disable if:** Working with sensitive content

### Auto Open File
- **Default:** Disabled  
- **What it does:** Opens saved file in default editor after transcription
- **Useful for:** Immediate review and editing
- **Enable if:** You always edit transcripts after recording

### Auto Metadata Generation
- **Default:** Enabled (if Ollama installed)
- **What it does:** AI generates filename, summary, and tags
- **Disable if:** Want manual control over metadata
- **Requirements:** Ollama must be installed and running

## 🧠 AI Analysis Settings

### Content Length Limits
- **Setting:** `OLLAMA_MAX_CONTENT_LENGTH`
- **Default:** 15,000 characters
- **What it does:** Maximum content size for AI processing
- **Large content:** Uses hierarchical chunking for files over 3,000 chars
- **Adjust if:** Getting timeouts or want to process larger files

### AI Analysis Features
- **Command:** `rec -g 000042` (AI analysis of transcripts)
- **Extracts:** Main themes, key questions, action items, narrative threads
- **Processing:** Hierarchical summarization for large content (30k+ characters)
- **Output:** Intelligent filename, comprehensive summary, relevant tags

### AI Timeout Settings  
- **Default:** 180 seconds (3 minutes)
- **What it does:** How long to wait for AI responses
- **Increase if:** Large files are timing out
- **Decrease if:** Want faster failures on problematic content

## ⚡ Performance Settings

### Basic Mode Settings
These are configured automatically for optimal performance:

#### Chunk Duration
- **Default:** 10 seconds
- **What it does:** How often you see transcription updates
- **Shorter (5s):** More frequent updates, higher CPU usage
- **Longer (15s):** Less frequent updates, lower CPU usage

#### No Speech Detection
- **Default:** 2 minutes
- **What it does:** Auto-stop when no speech detected
- **Shorter (30s):** Stops sooner during pauses
- **Longer (5min):** Waits longer during long pauses

### Detailed Mode Settings
Full control over performance parameters:

#### Advanced Chunking
- **Chunk Duration:** 5-30 seconds
- **Overlap:** How much chunks overlap for continuity
- **Worker Threads:** Number of parallel transcription workers
- **Buffer Size:** Audio buffer management

#### Voice Activity Detection
- **Sensitivity:** How sensitive to detect speech vs silence
- **Min Speech Duration:** Minimum time to consider as speech
- **Min Silence Duration:** Minimum silence before considering a pause

#### Memory Management  
- **Model Caching:** Keep models in memory vs reload each time
- **Chunk Buffering:** How many chunks to process simultaneously
- **Output Buffering:** How transcription is accumulated and displayed

## 🎤 Audio Settings

### Microphone Selection
```bash
python src/transcribe.py --list-devices
```
Shows available audio input devices:
```
Available audio devices:
  0: Built-in Microphone (default)
  1: USB Headset Microphone  
  2: Blue Yeti USB Microphone
```

Set preferred device in configuration or use:
```bash
python src/transcribe.py --device 2  # Use Blue Yeti
```

### Audio Quality Settings
- **Sample Rate:** 16kHz (optimized for Whisper)
- **Channels:** Mono (speech recognition works best with single channel)
- **Bit Depth:** 16-bit (sufficient for voice)

> **Note:** These are optimized automatically. Manual adjustment usually not needed.

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

### Custom Templates
Edit `templates.json` to customize output format:
```json
{
  "markdown": {
    "header": "---\ndate: {date}\nduration: {duration}\ntags: {tags}\nsummary: {summary}\n---\n\n# {title}\n\n",
    "content": "{transcript}"
  },
  "text": {
    "header": "Date: {date}\nDuration: {duration}\nTags: {tags}\nSummary: {summary}\n\n{title}\n\n",
    "content": "{transcript}"
  }
}
```

## 🔧 Environment Variables

Advanced users can set environment variables in `.env`:

```bash
# Whisper Settings
WHISPER_MODEL=small
WHISPER_LANGUAGE=auto

# Ollama Settings  
OLLAMA_MODEL=gemma3:4b
OLLAMA_TIMEOUT=30

# Output Settings
OUTPUT_FORMAT=md
OUTPUT_PATH=~/Documents/transcripts

# Auto Actions
AUTO_COPY=true
AUTO_OPEN=false
AUTO_METADATA=true

# Performance
CHUNK_DURATION=10
NO_SPEECH_TIMEOUT=120
WORKERS=2
```

## 🔄 Resetting Configuration

### Reset to Defaults
```bash
python src/transcribe.py --reset-config
```

### Reconfigure Everything
```bash
python src/transcribe.py --config
```

### Backup Settings
Your settings are stored in `.env`. Back it up:
```bash
cp .env .env.backup
```

## 🛠️ Troubleshooting Settings

### "Settings not saving"
- Check file permissions in the project directory
- Ensure `.env` file is writable
- Try running `rec-settings` as your user (not sudo)

### "Command not found after changing alias"
- Restart terminal: `source ~/.zshrc`
- Or use full path: `python /path/to/rejoice/src/transcribe.py`

### "AI features stopped working"
- Check Ollama: `ollama list`
- Test connection: `curl http://localhost:11434/api/version`
- Try different model: `ollama pull gemma3:4b`

### "Audio device not found"
- List devices: `python src/transcribe.py --list-devices`
- Check system audio preferences
- Try default device: Don't specify device number

---

**← [Back to Home](README.md)** | **Next: [Dependencies Guide →](DEPENDENCIES.md)**