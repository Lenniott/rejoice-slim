# 📖 How to Use Rejoice Slim

**← [Back to Home](README.md)**

> ⚠️ **macOS Only** - This guide is for macOS users. See [OS_AGNOSTIC_ROADMAP.md](Docs/Future_Thoughts/STANDALONE_APP_ANALYSIS.md) for other platforms.

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
🆔 Transcript ID: 000042
📁 Saved: ~/Documents/transcripts/project-timeline-discussion_22102025_000042.md
📋 Copied to clipboard
🚀 Open file? (y/n): 

💡 Reference this transcript with: rec -000042
```

## ⚙️ Available Commands

After installation, you get these commands:

### Quick Reference
```bash
# Recording
rec                         # Start recording
rec ID                      # Append to existing transcript

# Viewing
rec -l / --list            # List all transcripts
rec -v ID / --view ID      # View transcript content
rec --audio ID             # Show audio files for transcript
rec -o / --open-folder     # Open transcripts folder

# AI Analysis
rec -g ID / --genai ID     # AI analysis: themes, questions, actions
rec -g /path/to/file.md    # Analyze any text file
rec -f FILE / --format     # Format text into paragraphs

# Reprocessing
rec --reprocess ID         # Reprocess transcript audio
rec --reprocess-failed     # Process orphaned audio files

# Recovery
rec -ls / --list-sessions  # List interrupted sessions
rec -r / --recover         # Recover latest session

# Settings
rec -s / --settings        # Configure interactively
```

### 🤖 AI Analysis Features
```bash
# Analyze transcripts or any text file
rec -g 000042                    # Analyze transcript by ID
rec -g /path/to/file.md         # Analyze any text file
rec --genai /path/to/notes.txt  # Long form command

# AI extracts:
# - Main themes and narrative threads
# - Key questions asked during conversation  
# - Action items and decisions made
# - Intelligent filename suggestions
# - Relevant tags for categorization

# Uses hierarchical processing for large files:
# - Breaks content into ~2000 character chunks
# - Summarizes each chunk focusing on themes/questions/actions
# - Creates meta-summary from all chunks
# - Handles files up to 30k+ characters efficiently
```

### 📝 Text Formatting
```bash
# Format text files into meaningful paragraphs using AI
rec -f /path/to/file.txt        # Format text file
rec --format /path/to/notes.md  # Long form command

# How it works:
# - Breaks text into chunks (~1000 chars with 200 char overlap)
# - AI reformats each chunk into clear paragraphs
# - Preserves all original content (no summarization)
# - Supports thinking models (removes <think> tags)
# - Overwrites the original file with formatted version

# Perfect for:
# - Raw transcripts that need better paragraph structure
# - Stream-of-consciousness notes
# - Unformatted text dumps
# - Any text needing better readability

# Example workflow:
rec                             # Record meeting
rec -f ~/transcripts/000042*    # Format the transcript
rec -g 000042                   # Then analyze with AI
```

### 🔗 Appending to Existing Transcripts
```bash
# Create a new transcript
rec                   # Creates: meeting-notes_22102025_000001.md

# Append to an existing transcript by ID
rec 000042            # Record more audio and add to transcript 000042
rec 1                 # Works with short IDs too

# View and manage
rec -l                # List all transcripts with their IDs
rec -v 000042         # View transcript content
```

**How appending works:**
1. Shows preview of existing transcript content
2. Records new audio
3. Appends new transcription to the same file
4. Preserves all audio files linked to that transcript ID

### 🔍 Viewing & Managing Transcripts
```bash
rec -l / --list              # List all transcripts with IDs
rec -v ID / --view ID        # View transcript content by ID
rec --audio ID               # Show audio files for transcript ID
rec -o / --open-folder       # Open transcripts folder
```

### 🔄 Reprocessing Commands
```bash
rec --reprocess ID           # Reprocess all audio for transcript ID
                            # - Re-transcribes all audio files
                            # - Generates fresh AI summary
                            # - Creates new version by default

rec --reprocess ID --overwrite  # Overwrite existing transcript
                                # - Replaces original instead of creating new

rec --reprocess-failed       # Process orphaned audio files
                            # - Finds audio files without transcripts
                            # - Creates new transcripts for them
```

### 💾 Recovery Commands
```bash
rec -ls / --list-sessions    # List interrupted recording sessions
rec -r / --recover           # Recover latest interrupted session
rec -r ID / --recover ID     # Recover specific session by ID
```

**When to use recovery:**
- If recording was interrupted (crash, Ctrl+C, power loss)
- Audio is preserved in temporary session files
- Can transcribe the full audio even after interruption

### 🎛️ Recording Options
```bash
rec --verbose               # Enable detailed streaming transcription output
rec -ts / --timestamps      # Include timestamps in transcript output
rec --device N              # Use specific microphone (N = device number)
rec --copy / --no-copy     # Override clipboard auto-copy setting
rec --open / --no-open     # Override auto-open file setting
rec --metadata / --no-metadata  # Override AI metadata generation
```

### ⏱️ Timestamps Feature
```bash
rec -ts                     # Record with timestamps
rec --timestamps            # Long form command
```

**What you get:**
```markdown
[00:00] This is the first part of the recording at the beginning.

[00:15] Here's another segment that was detected after 15 seconds.

[01:30] This segment starts at one minute and thirty seconds.

[05:45] And this is near the end of the recording.
```

**How it works:**
- Timestamps use `[MM:SS]` format (or `[HH:MM:SS]` for recordings over an hour)
- Each timestamp marks the start of a segment detected by Whisper
- Segments are separated by blank lines for readability
- Works with the full transcription (not streaming quick transcript)
- Perfect for meetings, interviews, or any content where timing matters

## 📝 Understanding Output

### New ID-Based Transcript File
```markdown
---
id: '000042'
title: '000042'
created: '2025-10-22T14:30:15.123456'
status: raw
---

## 🎙️ Transcription

The main points we covered today were around the new feature timeline. 
We need to prioritize the user dashboard updates before the end of the quarter...

[Rest of your transcription here]
```

### YAML Frontmatter Explained (New Format)
- **🆔 id**: Unique 6-digit identifier (same as filename)
- **📝 title**: Simple title (currently same as ID)
- **📅 created**: ISO 8601 timestamp of when transcript was created
- **📊 status**: Processing status (raw, processed, etc.)

### Legacy Format (Still Readable)
Older transcripts may have the previous format with tags, summaries, and AI metadata. The system can read both formats seamlessly.

## 🎯 Best Practices

### For Better Transcription
- **Speak clearly** - Not too fast, not too quiet
- **Minimize background noise** - Close windows, turn off fans
- **Use a good microphone** - Built-in mics work, external mics are better
- **Pause between thoughts** - Helps with natural sentence breaks

### For Obsidian Users
- **Enable Obsidian integration** - Configure during setup or with `rec -s` → Casual → Configure Obsidian Integration
- **Vault selection** - Pick your vault root from a numbered directory list
- **Automatic Markdown** - Output format set to `md` when Obsidian integration is enabled
- **URI-based opening** - Files open directly in Obsidian using proper vault URIs
- **Use AI tags** - They automatically link to other notes
- **Enable auto-metadata** - Summaries help with note organization

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

### Advanced Usage

### Command Line Options
```bash
python src/transcribe.py --help                    # Show all options

# Short options (recommended)
python src/transcribe.py -s                        # Settings menu
python src/transcribe.py -l                        # List all transcripts  
python src/transcribe.py -v 000042                 # View transcript content
python src/transcribe.py -g 000042                 # AI analysis and tagging

# Long options (also available)  
python src/transcribe.py --settings                # Same as -s
python src/transcribe.py --list                    # Same as -l
python src/transcribe.py --view 000042             # Same as -v
python src/transcribe.py --genai 000042            # Same as -g

# Other options
python src/transcribe.py -000042                   # Append to transcript 000042
python src/transcribe.py --device 1                # Use specific audio device
```

### ID Management
```bash
# Create new transcript (gets next available ID)
python src/transcribe.py

# Work with existing transcripts
python src/transcribe.py --list                    # See all IDs
python src/transcribe.py --show 000001             # Read transcript
python src/transcribe.py -000001                   # Append to transcript

# ID format is always 6 digits: 000001, 000042, 999999
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

### "How do I find old transcripts?"
- **Use `rec --list`** to see all transcripts with IDs and dates
- **ID format**: New transcripts use descriptive-name_DDMMYYYY_000001.md format
- **Legacy format**: Older files keep their timestamp names
- **Both work**: You can reference and edit files in either format

### "How accurate is the transcription?"
- **Small model**: Very good for clear speech (~95% accuracy)
- **Large model**: Excellent for all conditions (~98% accuracy)  
- **Factors**: Audio quality, speaking clarity, background noise

### "Can I edit transcriptions?"
- **Yes!** All files are standard Markdown/text
- **Edit directly** in Obsidian, VS Code, or any text editor
- **Metadata preserved** in YAML frontmatter
- **ID system preserved** - editing doesn't affect referencing

### "Does it work offline?"
- **Yes!** After initial setup, no internet required
- **All processing** happens on your device
- **Ollama runs locally** - no cloud AI services

### "What about privacy?"
- **100% local** - audio never leaves your computer
- **No cloud services** - Whisper and Ollama run locally
- **No telemetry** - zero data collection
- **Open source** - verify the code yourself

## 🆔 ID-Based Transcript System

### How It Works
- **New transcripts** use smart naming: descriptive-name_DDMMYYYY_000001.md
- **Easy referencing** with short commands: `rec -000042`
- **YAML headers** contain ID, creation date, and metadata
- **Backward compatible** with legacy timestamp-based files

### Common ID Operations
```bash
# List all transcripts (shows both new ID and legacy formats)
rec --list

# View transcript content
rec --show 000042

# Append to existing transcript (records new audio and adds it)
rec -000042

# Create new transcript (gets next available ID automatically)
rec
```

### File Structure
```
transcripts/
├── meeting-notes_22102025_000001.md     # New ID format with smart naming
├── project-ideas_22102025_000002.txt    # Format depends on your settings  
├── interview_22102025_000003.md         # AI generates descriptive names
└── legacy_timestamp_20241020.md         # Old timestamp format (still accessible)
```

### Migration from Legacy
- **Existing files** remain unchanged and accessible
- **New recordings** use the ID system automatically  
- **No migration required** - both formats work together
- **Gradual transition** as you create new transcripts

## 🤖 AI Analysis Deep Dive

### What the AI Extracts
When you run `rec -g 000042`, the AI analyzes your transcript and extracts:

**📋 Key Themes & Topics**
- Main discussion topics and subjects
- Recurring themes throughout the conversation
- Central narrative threads

**❓ Questions & Inquiries**  
- Important questions asked during discussion
- Information requests and clarifications
- Decision points that need follow-up

**✅ Actions & Decisions**
- Action items assigned to individuals
- Decisions made during the meeting
- Next steps and commitments

**🏷️ Smart Tags**
- Relevant keywords and categories  
- Technical terms and concepts mentioned
- Project names and system references

### Hierarchical Processing
For large transcripts (3000+ characters), the AI uses advanced hierarchical processing:

1. **📑 Chunking**: Breaks content into ~2000 character overlapping chunks
2. **🔍 Chunk Analysis**: Each chunk summarized focusing on themes/questions/actions  
3. **🎯 Meta-Summary**: Combines all chunk summaries into final analysis
4. **⚡ Efficiency**: Handles transcripts up to 30,000+ characters reliably

### AI Output Example
```markdown
📁 Filename: NHS_Patient_Flag_Implementation  
📝 Summary: Discussion about implementing patient flags in NHS services, focusing on reasonable adjustments, data integration challenges, and supplier onboarding processes with September deadline.
🏷️ Tags: reasonable-adjustments, data-integration, supplier-onboarding, nhs-england, patient-flags
```

### Configuration
- **Model Selection**: Change AI model (`rec -s` → Core → Change Ollama Model)
- **Content Limits**: Adjust max content length (`rec -s` → Advanced → Max AI Content Length)
- **Timeout Settings**: Configure AI timeout (`rec -s` → Advanced → Change Ollama Timeout)
- **Auto-metadata**: Toggle automatic AI analysis (`rec -s` → Casual → Toggle Auto Metadata)

## 📒 Obsidian Integration

### Setting Up Obsidian Integration

Rejoice Slim has sophisticated Obsidian vault integration that ensures transcripts open directly in Obsidian with proper vault context.

#### During Initial Setup
```
Step 1: Enter save path
/Users/you/Documents/Obsidian/MyVault/Notes/Transcripts

Step 2: Integrate with Obsidian? [y/n]: y

Step 3: Select your vault from the path:
1. Users
2. you
3. Documents
4. Obsidian
5. MyVault  ← Your vault
6. Notes
7. Transcripts

Which directory is your Obsidian vault? [1-7, or 0 to cancel]: 5

✓ Selected vault: MyVault
✓ Output format set to 'md' (required for Obsidian integration)
```

#### Configuring Later
```bash
rec -s                           # Open settings
# Choose: Casual Settings → Configure Obsidian Integration
```

### How It Works

**Smart Vault Detection**
- Parses your save path into directory components
- Shows you each directory as a numbered option
- You select which one is your vault root
- No guessing, no auto-detection issues

**URI-Based Opening**
- Uses `obsidian://open?vault=YourVault&file=path/to/file.md`
- Opens files with full vault context
- Works with iCloud-synced vaults
- Falls back to default app if file is outside vault

**Automatic Markdown**
- When Obsidian integration is enabled, output format is set to `md`
- You won't be asked to choose format during setup
- Ensures compatibility with Obsidian

### Benefits

✅ **Reliable Opening** - No more "file not found" errors
✅ **Vault Context** - Files open with full Obsidian features (links, tags, graph view)
✅ **iCloud Support** - Works with iCloud-synced Obsidian vaults
✅ **Easy Reconfiguration** - Change vault selection anytime via settings
✅ **Fallback Handling** - If file is outside vault, opens with default app

### Example Workflow

```bash
# 1. Setup with Obsidian integration enabled
python configure.py
# → Save path: /Users/you/.../MyVault/Transcripts
# → Obsidian? y
# → Select vault: MyVault

# 2. Record a transcript
rec
# → File saved: MyVault/Transcripts/meeting-notes_25122025_000001.md

# 3. Open automatically
# → Opens directly in Obsidian with vault context
# → Can see backlinks, tags, graph connections
```

### Troubleshooting Obsidian Integration

**Files not opening in Obsidian?**
- Verify save path is inside your vault: `rec -s` → Core → View Settings
- Check vault path: Look for "Obsidian Vault: ✅ VaultName" in settings summary
- Reconfigure if needed: `rec -s` → Casual → Configure Obsidian Integration

**Want to change vault?**
- Run `rec -s` → Casual → Configure Obsidian Integration
- Select new vault from directory list
- Previous transcripts remain unchanged

**Want to disable Obsidian integration?**
- Run `rec -s` → Casual → Configure Obsidian Integration
- Answer "n" when asked "Integrate with Obsidian?"

## 🔧 Troubleshooting

### "No audio detected"
- Check microphone permissions in System Preferences
- Try a different audio device: `python src/transcribe.py --list-devices`
- Test microphone in other apps first

### "Transcription is slow"
- Use a smaller Whisper model: `rec -s` → Core → Change Whisper Model (choose 'tiny' or 'base')
- Check available RAM (larger models need more memory)
- Close other resource-intensive applications

### "AI features not working"
- Check if Ollama is running: `ollama list`
- Try a different model: `ollama pull gemma3:4b`
- Restart Ollama: `ollama serve`
- Check AI settings: `rec -s` → AI settings

### "AI analysis fails on large files"
- Content automatically truncated at 32,000 chars (configurable in settings)
- Check Ollama model has enough memory for processing
- Try a smaller, faster model like `gemma3:4b`
- Adjust max content length: `rec -s` → Advanced → Change Max AI Content Length

## 📋 Quick Reference Card

### Essential Commands
```bash
rec           # Start new recording
rec -l        # List all transcripts  
rec -v 000042 # View transcript content
rec -g 000042 # AI analysis (themes, questions, actions)
rec -s        # Settings menu
rec -000042   # Append to existing transcript
```

### AI Analysis Options  
```bash
rec -g 000042              # Analyze transcript by ID
rec -g /path/to/file.md    # Analyze any text file  
rec --genai 000042         # Same as -g (long form)
```

### File Management
```bash  
rec --list                 # List all transcripts (long form)
rec --view 000042          # View content (long form)
rec -o                     # Open transcripts folder
```

### Advanced Options
```bash
rec --device 2             # Use specific microphone
rec --language es          # Set language (Spanish)
rec --no-copy              # Don't copy to clipboard
rec --no-metadata          # Skip AI processing
```

---

**← [Back to Home](README.md)** | **Next: [Settings Guide →](SETTINGS.md)**