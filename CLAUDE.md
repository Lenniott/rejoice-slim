# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rejoice Slim is a local, privacy-first voice transcription tool for macOS that integrates with Obsidian. It converts spoken audio into searchable markdown transcripts without sending data to cloud services.

**Core Philosophy**: 100% offline operation - all processing happens locally using Whisper (speech-to-text) and Ollama (AI analysis).

## Essential Commands

### Development Setup
```bash
# Full automated installation
curl -fsSL https://raw.githubusercontent.com/benjamayden/rejoice-slim/main/setup.sh | bash

# Manual setup (if needed)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 configure.py
```

### Running the Application
```bash
# Via alias (normal usage after installation)
rec                    # Start recording
rec -ts                # Start recording with timestamps
rec -l                 # List transcripts
rec -v 000042          # View transcript
rec -g 000042          # AI analysis
rec -s                 # Settings menu

# Direct Python execution (for development/debugging)
python src/transcribe.py
python src/transcribe.py -ts
python src/transcribe.py -l
python src/transcribe.py -v 000042
```

### Configuration
```bash
# Initial setup wizard
python configure.py

# Reconfigure anytime
rec -s

# Direct .env editing (use with caution)
# Located at: .env (600 permissions)
```

### No Traditional Build/Test System
- No build step required (Python interpreter runs directly)
- No formal test suite currently (manual integration testing)
- Settings validation happens at runtime via `settings.py`

## High-Level Architecture

### Core Design Pattern: Modular Separation

The project was refactored (Nov 2025) from a 2,064-line monolithic file into focused modules:

```
transcribe.py (809 lines)      → Recording loop + CLI routing
commands.py (568 lines)        → All user-facing command implementations
settings.py (632 lines)        → Interactive configuration UI
summarization_service.py       → AI analysis with factory pattern
```

**Key Architectural Decision**: Configuration is passed as parameters (not globals), making each module independently testable.

### Real-Time Streaming Pipeline

This is the most complex part of the architecture and requires reading multiple files to understand:

```
Audio Input (sounddevice)
    ↓
CircularAudioBuffer (audio_buffer.py)
    - Thread-safe ring buffer (default: 5 minutes at 16kHz)
    - Fixed memory footprint (~150MB)
    - Supports concurrent write + segment extraction
    ↓
VolumeSegmenter (volume_segmenter.py)
    - Real-time audio analysis (1-second RMS windows)
    - Detects natural speech boundaries via volume drops
    - Configurable segment duration (30/60/90s min/target/max)
    ↓
WhisperEngine (whisper_engine.py)
    - faster-whisper wrapper (4x faster than OpenAI version)
    - Transcribes segments in near real-time
    ↓
QuickTranscriptAssembler (quick_transcript.py)
    - Assembles segments as they complete
    - Manages clipboard integration
    - Tracks status with callbacks
    ↓
User sees live transcript (streaming feedback)
```

**Thread Safety**: All components use `threading.RLock` to handle concurrent recording + transcription.

### File Management System

**ID-Based Naming**: Replaces timestamp-based naming with sequential IDs for easier reference.

```
Format: {ID}_{DDMMYYYY}_{descriptive-name}.md
Example: 000042_25122025_meeting-notes.md
```

**Key Components**:
- `TranscriptIDGenerator` (id_generator.py) - Sequential ID generation, supports flexible matching
- `TranscriptFileManager` (transcript_manager.py) - File creation, appending, listing
- `TranscriptHeader` (file_header.py) - YAML frontmatter from template_config.yaml
- `AudioFileManager` (audio_manager.py) - Tracks audio files per transcript

**YAML Template System**: The `template_config.yaml` defines frontmatter structure with placeholders:
- `{{id}}` - 6-digit transcript ID
- `{{created}}` - ISO timestamp
- `{{archive_by}}` - 30 days from creation

### AI Summarization Architecture

**Factory Pattern**: `get_summarizer()` in [summarization_service.py](src/summarization_service.py) centralizes Ollama instantiation:
- Loads config from `.env`
- Eliminates duplicate setup code
- Makes testing easier

**Hierarchical Processing**: For large content (30k+ characters):
1. Breaks content into ~2000 character overlapping chunks
2. Summarizes each chunk (themes, questions, actions)
3. Creates meta-summary from all chunks
4. Extracts tags and generates smart filenames

**Security**: `_validate_localhost_url()` enforces localhost-only Ollama URLs to prevent data exfiltration.

### Obsidian Integration

[obsidian_utils.py](src/obsidian_utils.py) provides:
- URI-based file opening (`obsidian://open?vault=...&file=...`)
- Interactive vault path selection (numbered directory list)
- Automatic markdown format enforcement when enabled
- Fallback to default app if file outside vault

Configuration stored in `.env`:
```bash
OBSIDIAN_ENABLED=true/false
OBSIDIAN_VAULT_PATH=/path/to/vault
```

### Recovery System

[safety_net.py](src/safety_net.py) handles interrupted recordings:
- Tracks sessions with metadata in temp directory
- Preserves audio files from crashes/Ctrl+C
- `list_recovery_sessions()` - Show interrupted sessions
- `recover_session()` - Restore and fully transcribe

## Entry Points & Workflows

### Main Entry Point
[transcribe.py:main()](src/transcribe.py) - Parses CLI arguments and routes to:
- Recording: `record_audio_streaming()`
- Commands: Functions in [commands.py](src/commands.py)
- Settings: `settings_menu()` from [settings.py](src/settings.py)

### Common Code Paths

**Recording Flow**:
```
rec → main() → record_audio_streaming()
  → CircularAudioBuffer starts capturing
  → VolumeSegmenter detects segments
  → WhisperEngine transcribes in background
  → QuickTranscriptAssembler shows live results
  → User presses Enter or auto-stop triggers
  → TranscriptFileManager saves to disk
  → Optional: clipboard copy, Obsidian open, AI metadata
```

**AI Analysis Flow**:
```
rec -g 000042 → commands.summarize_file()
  → get_summarizer() factory creates service
  → SummarizationService.process_file()
  → Hierarchical chunking if needed
  → Ollama API call (localhost:11434)
  → Extract themes, questions, actions, tags
  → Generate smart filename
  → Update YAML frontmatter
```

**Settings Flow**:
```
rec -s → settings_menu()
  → Display interactive menu (5 sections)
  → Route to sub-menus based on choice
  → update_env_setting() modifies .env file
  → Persist changes with 600 permissions
```

## Configuration System

### Environment Variables (.env)

The `.env` file is the single source of truth for configuration. Key variables:

**Core Settings**:
```bash
SAVE_PATH=/path/to/transcripts        # Where transcripts are saved
WHISPER_MODEL=small                   # tiny/base/small/medium/large
WHISPER_LANGUAGE=auto                 # auto or language code
OUTPUT_FORMAT=md                      # md/txt
```

**Streaming Settings** (affects CircularAudioBuffer + VolumeSegmenter):
```bash
STREAMING_BUFFER_SIZE_SECONDS=300           # Ring buffer size (5 min)
STREAMING_MIN_SEGMENT_DURATION=30           # Min segment length
STREAMING_TARGET_SEGMENT_DURATION=60        # Target segment length
STREAMING_MAX_SEGMENT_DURATION=90           # Max segment length
```

**AI Settings** (affects get_summarizer() factory):
```bash
OLLAMA_MODEL=gemma3:4b                      # LLM model name
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_TIMEOUT=180                          # Seconds
OLLAMA_MAX_CONTENT_LENGTH=32000             # Characters
```

**Auto-Actions**:
```bash
AUTO_COPY=false          # Clipboard copy after recording
AUTO_OPEN=false          # Open file after recording
AUTO_METADATA=false      # AI metadata generation
```

**Obsidian**:
```bash
OBSIDIAN_ENABLED=false
OBSIDIAN_VAULT_PATH=/path/to/vault
```

**User Customization**:
```bash
COMMAND_NAME=rec                           # Custom CLI alias
DEFAULT_MIC_DEVICE=-1                     # -1 = default, or device number
```

### Loading Flow
1. [transcribe.py](src/transcribe.py) calls `load_dotenv()` at startup
2. Extracts values using `os.getenv()` with defaults
3. Passes config to components as parameters
4. Factory functions (like `get_summarizer()`) reload config when called

## Important Implementation Details

### Thread Safety Pattern
All streaming components use the same pattern:
```python
import threading

class Component:
    def __init__(self):
        self._lock = threading.RLock()

    def operation(self):
        with self._lock:
            # Critical section
```

Used in: `CircularAudioBuffer`, `VolumeSegmenter`, `QuickTranscriptAssembler`

### Signal Handling
[transcribe.py:_global_signal_handler()](src/transcribe.py) manages Ctrl+C:
- Graceful shutdown of recording
- Audio buffer finalization
- Session file preservation for recovery

### Clipboard Integration
Uses `pyperclip` in [quick_transcript.py](src/quick_transcript.py):
- Auto-copies assembled transcript
- Respects `AUTO_COPY` setting
- Falls back gracefully if clipboard unavailable

### File Header Generation
[file_header.py](src/file_header.py) loads `template_config.yaml` and replaces placeholders:
```python
def generate_header(transcript_id: str) -> str:
    # Loads template_config.yaml
    # Replaces {{id}}, {{created}}, {{archive_by}}
    # Returns YAML frontmatter as string
```

## Technology Stack

**Core Dependencies** (see [requirements.txt](requirements.txt)):
- `faster-whisper>=1.1.0` - Speech-to-text (4x faster than OpenAI version)
- `sounddevice>=0.5.0` - Microphone recording
- `scipy>=1.11.0` - Audio processing
- `numpy>=1.24.0,<2.0.0` - Audio tensor operations
- `python-dotenv>=1.0.1` - Configuration loading
- `requests>=2.32.0` - Ollama HTTP communication
- `pyperclip>=1.9.0` - Clipboard integration
- `PyYAML>=6.0.2` - YAML frontmatter

**External Dependencies**:
- **PortAudio** - Required by sounddevice (install via Homebrew on macOS)
- **Ollama** - Optional AI features (localhost server on port 11434)

**Python Version**: 3.9+ required (scipy 1.14+ requires 3.10+)

**Platform**: Currently macOS only (see [OS_AGNOSTIC_ROADMAP.md](OS_AGNOSTIC_ROADMAP.md))

## Recent Refactoring (Nov 2025)

The codebase underwent major cleanup:
- **Before**: Monolithic 2,064-line [transcribe.py](src/transcribe.py)
- **After**: 809 lines (61% reduction)
- **Extracted**: 632 lines → [settings.py](src/settings.py)
- **Extracted**: 568 lines → [commands.py](src/commands.py)
- **Deleted**: Dead code (unused chunker, worker modules)
- **Simplified**: Factory function for AI service

This means older commits may reference code that no longer exists.

## Common Gotchas

### 1. Factory Function Pattern
**Always use** `get_summarizer()` instead of instantiating `SummarizationService` directly:
```python
# CORRECT
from summarization_service import get_summarizer
summarizer = get_summarizer()

# WRONG - duplicates config loading
from summarization_service import SummarizationService
summarizer = SummarizationService(...)  # Don't do this
```

### 2. Configuration Parameters
Commands in [commands.py](src/commands.py) receive config as parameters, not globals:
```python
def list_transcripts(save_path: str, output_format: str):
    # Uses parameters, not global variables
```

This makes testing easier but means config must be threaded through call chains.

### 3. Legacy Format Support
The system supports both ID-based and legacy timestamp-based files:
- **New**: `000042_25122025_meeting-notes.md`
- **Legacy**: `meeting-notes_20241225_143015.md`

ID matching is flexible (e.g., `"42"` matches `"000042"`).

### 4. Localhost-Only AI
The `_validate_localhost_url()` function in [summarization_service.py](src/summarization_service.py) **rejects** non-localhost URLs:
```python
# This will fail validation
OLLAMA_API_URL=http://api.example.com/generate  # Rejected!

# This is the only allowed pattern
OLLAMA_API_URL=http://localhost:11434/api/generate  # OK
```

This enforces the "no cloud" privacy guarantee.

### 5. Streaming Buffer Size
The `STREAMING_BUFFER_SIZE_SECONDS` setting affects memory usage:
- Default: 300 seconds (5 min) = ~150MB
- Formula: `buffer_size * sample_rate * bytes_per_sample`
- Larger buffers = more memory but better recovery from slow transcription

### 6. Timestamp Formatting
The timestamp feature (`-ts` / `--timestamps`) uses a two-step process:
1. **Segment extraction**: `format_transcript_with_timestamps()` in [transcribe.py](src/transcribe.py) adds `[MM:SS]` markers from Whisper segment data
2. **Post-processing**: `post_format_timestamps()` uses regex `(?=\[[0-9]+:[0-9]+\])` to split text and ensure proper newlines

This dual approach ensures timestamps are correctly formatted regardless of how the text is assembled (streaming vs full transcription).

**Implementation locations**:
- CLI argument: [transcribe.py:1317-1318](src/transcribe.py#L1317-L1318)
- Formatting functions: [transcribe.py:144-204](src/transcribe.py#L144-L204)
- Full transcription: [transcribe.py:962-966](src/transcribe.py#L962-L966)
- Session file transcription: [transcribe.py:253-256](src/transcribe.py#L253-L256)

## When Making Changes

### Adding a New Command
1. Add function to [commands.py](src/commands.py)
2. Accept config as parameters (not globals)
3. Add CLI argument to [transcribe.py:main()](src/transcribe.py)
4. Update [USAGE.md](USAGE.md) with examples

### Modifying Streaming Pipeline
1. Understand the full flow: `CircularAudioBuffer` → `VolumeSegmenter` → `QuickTranscriptAssembler`
2. Maintain thread safety with `threading.RLock`
3. Test with long recordings (>5 minutes)
4. Verify recovery system still works

### Adding a Settings Option
1. Add to `.env` via [settings.py:update_env_setting()](src/settings.py)
2. Add menu option in appropriate settings section
3. Document in [SETTINGS.md](SETTINGS.md)
4. Add validation if needed

### Changing AI Behavior
1. Modify prompts in `prompts.json`
2. Update hierarchical processing logic in [summarization_service.py](src/summarization_service.py)
3. Test with large files (30k+ characters)
4. Ensure localhost validation remains strict

## Documentation References

- [README.md](README.md) - Overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed technical documentation
- [USAGE.md](USAGE.md) - Complete user guide
- [SETTINGS.md](SETTINGS.md) - Configuration reference
- [INSTALLATION.md](INSTALLATION.md) - Setup guide
- [DEPENDENCIES.md](DEPENDENCIES.md) - Package details
- [OS_AGNOSTIC_ROADMAP.md](OS_AGNOSTIC_ROADMAP.md) - Future Linux/Windows support
