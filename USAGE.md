# 📖 How to Use Rejoice

**← [Back to Home](README.md)**

## 🎤 Basic Recording

### Start Recording
```bash
rec
```
- **Press Enter** when you want to stop recording
- **Real-time transcription** - See your words appear as you speak
- **Smart auto-stop** - Automatically stops when no speech detected for 2+ minutes

### What You'll See
```
🔴 Recording... (00:01:23) 
🎯 Voice detected: Speaking...

Live transcription:
Today we discussed the new project timeline and the key milestones 
we need to hit for the Q4 release...

[Press Enter to stop, or wait for auto-stop]
```

### After Recording
```
✅ Transcription complete! 
📁 Saved: ~/Documents/transcripts/project-timeline-discussion-2024-10-20.md
📋 Copied to clipboard
🚀 Open file? (y/n): 
```

## ⚙️ Available Commands

After installation, you get these commands:

### `rec` - Start Recording
```bash
rec                    # Basic recording with all features
rec --no-ai           # Skip AI features (faster)  
rec --format txt      # Save as plain text instead of Markdown
rec --output ~/Notes  # Save to specific directory
```

### `rec-settings` - Configure Settings
```bash
rec-settings
```
Interactive menu to change:
- **Whisper model** (tiny/base/small/medium/large)
- **Ollama model** (gemma3:4b, llama3, qwen3:0.6b, etc.)
- **Output format** (Markdown or plain text)
- **Save location** (your Obsidian vault, Documents, etc.)
- **Auto-actions** (copy to clipboard, auto-open files)

### `open-transcripts` - View Your Files
```bash
open-transcripts
```
Opens your transcripts folder in Finder/Explorer.

## 📝 Understanding Output

### Example Transcript File
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

The main points we covered today were around the new feature timeline. 
We need to prioritize the user dashboard updates before the end of the quarter...

[Rest of your transcription here]
```

### YAML Frontmatter Explained
- **📅 date**: When the recording was made
- **⏱️ duration**: Length of the recording  
- **🏷️ tags**: AI-generated relevant tags (spaces converted to hyphens for Obsidian)
- **📝 summary**: AI-generated content summary
- **🤖 ai_model/whisper_model**: Which AI models were used

## 🎯 Best Practices

### For Better Transcription
- **Speak clearly** - Not too fast, not too quiet
- **Minimize background noise** - Close windows, turn off fans
- **Use a good microphone** - Built-in mics work, external mics are better
- **Pause between thoughts** - Helps with natural sentence breaks

### For Obsidian Users
- **Set save path to your vault** - During setup or with `rec-settings`
- **Use AI tags** - They automatically link to other notes
- **Enable auto-metadata** - Summaries help with note organization
- **Choose Markdown format** - Better integration with Obsidian features

### For Meeting Notes
- **State the meeting topic** at the start - Helps AI generate better filenames
- **Mention attendees** if relevant
- **Speak action items clearly** - "Action item: John to follow up on..."
- **Use the auto-stop feature** - Let it stop when discussion naturally ends

## 🔄 Real-time Features

### During Recording
- **🔴 Recording indicator** - Shows when actively recording
- **⏱️ Live timer** - Displays recording duration  
- **📝 Live transcription** - See words appear in real-time
- **🎯 Voice detection** - Visual feedback when speech is detected
- **⏹️ Smart auto-stop** - Stops automatically when you're done speaking

### Processing Options
- **📋 Copy to clipboard** - Paste transcript anywhere
- **📂 Auto-open file** - Review and edit immediately
- **🤖 AI enhancement** - Smart filenames and metadata (requires Ollama)

## 🛠️ Advanced Usage

### Command Line Options
```bash
python src/transcribe.py --help                    # Show all options
python src/transcribe.py --config                  # Configure settings
python src/transcribe.py --model large             # Use specific Whisper model
python src/transcribe.py --language en             # Force English transcription
python src/transcribe.py --device 1                # Use specific audio device
python src/transcribe.py --no-ai                   # Skip AI features
python src/transcribe.py --output ~/MyNotes        # Custom output directory
```

### Multiple Microphones
If you have multiple audio devices:
```bash
python src/transcribe.py --list-devices  # Show available microphones
python src/transcribe.py --device 2      # Use device #2
```

### Different Languages
```bash
python src/transcribe.py --language es   # Spanish
python src/transcribe.py --language fr   # French  
python src/transcribe.py --language de   # German
# Or leave as 'auto' for automatic detection
```

## ❓ Common Questions

### "How accurate is the transcription?"
- **Small model**: Very good for clear speech (~95% accuracy)
- **Large model**: Excellent for all conditions (~98% accuracy)  
- **Factors**: Audio quality, speaking clarity, background noise

### "Can I edit transcriptions?"
- **Yes!** All files are standard Markdown/text
- **Edit directly** in Obsidian, VS Code, or any text editor
- **Metadata preserved** in YAML frontmatter

### "Does it work offline?"
- **Yes!** After initial setup, no internet required
- **All processing** happens on your device
- **Ollama runs locally** - no cloud AI services

### "What about privacy?"
- **100% local** - audio never leaves your computer
- **No cloud services** - Whisper and Ollama run locally
- **No telemetry** - zero data collection
- **Open source** - verify the code yourself

## 🔧 Troubleshooting

### "No audio detected"
- Check microphone permissions in System Preferences
- Try a different audio device: `python src/transcribe.py --list-devices`
- Test microphone in other apps first

### "Transcription is slow"
- Use a smaller Whisper model: `rec-settings` → choose 'tiny' or 'base'
- Check available RAM (larger models need more memory)
- Close other resource-intensive applications

### "AI features not working"
- Check if Ollama is running: `ollama list`
- Try a different model: `ollama pull gemma3:4b`
- Disable AI features if needed: `rec --no-ai`

---

**← [Back to Home](README.md)** | **Next: [Settings Guide →](SETTINGS.md)**