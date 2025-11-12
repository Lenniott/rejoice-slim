# Architecture Documentation

## Overview

Rejoice is a modular voice transcription system with real-time streaming capabilities and AI-powered analysis. The codebase follows a clean separation of concerns with distinct modules for recording, transcription, commands, and settings.

## Project Structure

```
rejoice/
├── src/
│   ├── transcribe.py           # Main entry point & recording loop (809 lines)
│   ├── commands.py              # CLI command handlers (568 lines)
│   ├── settings.py              # Settings menu system (632 lines)
│   ├── summarization_service.py # AI analysis & factory function
│   ├── audio_buffer.py          # Circular buffer for audio
│   ├── volume_segmenter.py      # Real-time audio segmentation
│   ├── quick_transcript.py      # Streaming transcription assembly
│   ├── background_enhancer.py   # Post-processing enhancement
│   ├── transcript_manager.py    # File & ID management
│   ├── audio_manager.py         # Audio file handling
│   ├── file_header.py           # Markdown header generation
│   ├── id_generator.py          # Transcript ID system
│   ├── loading_indicator.py     # UI loading animations
│   └── safety_net.py            # Session recovery system
├── configure.py                 # Initial setup wizard
├── setup.sh                     # Installation script
└── uninstall.sh                 # Cleanup script
```

## Core Modules

### 1. transcribe.py - Main Entry Point (809 lines)
**Responsibility:** Recording loop, signal handling, CLI argument parsing

**Key Functions:**
- `record_audio_streaming()` - Main recording loop with real-time transcription
- `transcribe_session_file()` - Recovery transcription for interrupted sessions
- `deduplicate_transcript()` - Text cleanup utility
- `handle_post_transcription_actions()` - Post-recording workflow
- `_global_signal_handler()` - Ctrl+C handling
- `main()` - CLI entry point and argument routing

**Design:** Focused on core recording functionality, delegates all commands to `commands.py` and settings to `settings.py`.

### 2. commands.py - CLI Commands (568 lines)
**Responsibility:** All user-facing command implementations

**Functions:**
- `open_transcripts_folder()` - Open save location
- `list_transcripts()` - Display all transcripts with IDs
- `show_transcript()` - View transcript content
- `show_audio_files()` - List associated audio files
- `append_to_transcript()` - Add to existing transcript
- `summarize_file()` - AI analysis of transcript or file
- `reprocess_transcript_command()` - Re-transcribe audio files
- `reprocess_failed_command()` - Process orphaned audio
- `list_recovery_sessions()` - Show interrupted sessions
- `recover_session()` - Restore interrupted recording

**Design:** Each command receives configuration parameters from `transcribe.py`, making them testable and modular.

### 3. settings.py - Configuration System (632 lines)
**Responsibility:** Interactive settings menu and .env management

**Functions:**
- `settings_menu()` - Main menu interface
- `transcription_settings()` - Whisper model & language
- `output_settings()` - Format, path, auto-actions
- `ai_settings()` - Ollama configuration
- `audio_settings()` - Microphone selection
- `advanced_performance_settings()` - Streaming & buffer config
- `command_settings()` - Custom command name
- `uninstall_settings()` - Removal wizard
- `update_env_setting()` - .env file updates
- `list_audio_devices()` - Device enumeration

**Design:** Self-contained menu system with no dependencies on recording logic.

### 4. summarization_service.py - AI Analysis
**Responsibility:** Ollama integration, hierarchical summarization, metadata generation

**Key Features:**
- `get_summarizer()` - Factory function (loads config from .env)
- `SummarizationService` - Main AI service class
- Hierarchical processing for large content (30k+ characters)
- Smart filename generation from content
- Tag extraction and frontmatter generation
- Conversation detection and formatting

**Design Pattern:** Factory function centralizes configuration, eliminating duplicate instantiations.

## Streaming Transcription Architecture

### Real-time Pipeline

```
Audio Input → CircularAudioBuffer → VolumeSegmenter → SegmentProcessor
                                            ↓
                                   QuickTranscriptAssembler
                                            ↓
                                  Immediate Transcript (5-30s)
                                            ↓
                                   BackgroundEnhancer
                                            ↓
                                  Enhanced Transcript (full quality)
```

### Key Components:

**CircularAudioBuffer** (`audio_buffer.py`)
- Fixed-size ring buffer (default 5 minutes)
- Thread-safe audio storage
- Segment extraction by timestamp

**VolumeSegmenter** (`volume_segmenter.py`)
- Real-time audio analysis
- Dynamic segment detection (30-90s chunks)
- Volume-based speech detection
- Pause detection for natural breaks

**QuickTranscriptAssembler** (`quick_transcript.py`)
- Manages streaming transcription results
- Assembles segments as they complete
- Dynamic timeout calculation
- Clipboard integration
- Status tracking and callbacks

**BackgroundEnhancer** (`background_enhancer.py`)
- Queue-based enhancement system
- Full audio transcription (higher quality)
- AI metadata generation
- Cleanup of temporary files

## File Management System

### ID-Based System
- Format: `{ID}_{DDMMYYYY}_{descriptive-name}.md`
- Example: `000042_12112025_meeting-notes.md`
- Supports both full ID and short reference

### TranscriptFileManager (`transcript_manager.py`)
- File creation with unique IDs
- Metadata parsing and updates
- Conflict resolution
- Legacy format support

### AudioFileManager (`audio_manager.py`)
- Audio file tracking per transcript
- Cleanup and validation
- Duration calculation
- Storage optimization

## Configuration System

### Environment Variables (.env)
```bash
# Core Settings
SAVE_PATH=/path/to/transcripts
OUTPUT_FORMAT=md
WHISPER_MODEL=small
WHISPER_LANGUAGE=auto

# AI Settings
OLLAMA_MODEL=gemma3:4b
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_TIMEOUT=180
OLLAMA_MAX_CONTENT_LENGTH=32000

# Auto-Actions
AUTO_COPY=false
AUTO_OPEN=false
AUTO_METADATA=true

# Streaming Settings
STREAMING_BUFFER_SIZE_SECONDS=300
STREAMING_MIN_SEGMENT_DURATION=30
STREAMING_TARGET_SEGMENT_DURATION=60
STREAMING_MAX_SEGMENT_DURATION=90

# Audio
DEFAULT_MIC_DEVICE=-1
SILENCE_DURATION_SECONDS=120
SAMPLE_RATE=16000

# Command
COMMAND_NAME=rec
```

## Error Handling & Recovery

### SafetyNetManager (`safety_net.py`)
- Session file tracking
- Crash recovery
- Audio preservation
- Graceful degradation

### Recovery Flow
1. Interrupted recording detected
2. Session file preserved with metadata
3. User can list sessions: `rec --list-sessions`
4. Recover with: `rec --recover {session_id}`
5. Full transcription from saved audio

## Testing Strategy

### Integration Tests
Tests verify end-to-end functionality:
- ✅ `rec --list` - Transcript listing
- ✅ `rec --view ID` - Content viewing
- ✅ `rec --audio ID` - Audio file listing
- ✅ `rec --settings` - Settings menu
- ✅ `rec --open-folder` - Folder opening

### Manual Testing
Recording workflows:
- Basic recording with auto-stop
- Append to existing transcript
- Recovery from interruption
- AI metadata generation

## Performance Characteristics

### Memory Usage
- Circular buffer: ~150MB (5 min @ 16kHz mono)
- Whisper model: 400MB-10GB depending on model size
- Ollama: 1-8GB depending on model

### Latency
- Quick transcript: 5-30 seconds after recording
- Enhanced transcript: 30-120 seconds background
- AI metadata: 10-60 seconds (Ollama dependent)

### Throughput
- Real-time factor: ~0.1-0.3x (Whisper processes faster than real-time)
- Segment processing: Parallel, non-blocking
- Background enhancement: Queue-based, no user blocking

## Design Principles

### 1. Modularity
- Each module has a single, clear responsibility
- Commands separated from recording logic
- Settings isolated from core functionality

### 2. Factory Pattern
- `get_summarizer()` centralizes AI service creation
- Loads configuration from environment
- Eliminates duplicate instantiation code

### 3. Separation of Concerns
- **transcribe.py**: Recording & CLI routing
- **commands.py**: Command implementations
- **settings.py**: Configuration management
- **summarization_service.py**: AI analysis

### 4. Testability
- Commands receive parameters (not globals)
- Functions are pure where possible
- Dependencies injected explicitly

### 5. User Experience
- Quick feedback (streaming transcription)
- Background processing (no blocking)
- Graceful degradation (fallbacks)
- Recovery mechanisms (interrupted sessions)

## Future Enhancements

### Potential Improvements
1. **Testing**: Unit tests for core modules
2. **Streaming**: Websocket-based live transcription viewer
3. **Multi-language**: Automatic language detection
4. **Cloud Sync**: Optional backup to personal storage
5. **Plugins**: Extension system for custom post-processing
6. **Web UI**: Browser-based dashboard
7. **Mobile**: Companion app for remote recording

## Code Cleanup History

### Refactoring (Nov 2025)
- **Before**: 2,064 lines in transcribe.py (monolithic)
- **After**: 809 lines in transcribe.py (61% reduction)
- **Extracted**: 632 lines → settings.py
- **Extracted**: 568 lines → commands.py
- **Deleted**: Dead code (audio_chunker, transcription_worker, etc.)
- **Simplified**: Factory function for AI service

See `CLEANUP_SUMMARY.md` for detailed phase-by-phase breakdown.

## Contributing

When adding features:
1. Determine correct module (transcribe, commands, settings, or new)
2. Add parameters instead of using globals when possible
3. Update this document with architectural changes
4. Add integration tests for new commands
5. Update user documentation (USAGE.md)

## Questions?

For implementation details:
- See individual module docstrings
- Check function signatures in commands.py
- Review streaming pipeline in quick_transcript.py
- Examine AI processing in summarization_service.py

