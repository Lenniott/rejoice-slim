<!-- f5a8f0ba-a864-4459-8d27-8edf0fae673f 29227cb8-dd0c-41f4-8c21-d4631a67d4d6 -->
# Code Cleanup and Refactoring Plan

## Phase 1: Delete Dead Code Files

Remove completely unused files that were replaced by the streaming system:

- Delete [`src/audio_chunker.py`](src/audio_chunker.py) - old chunking system
- Delete [`src/transcription_worker.py`](src/transcription_worker.py) - old worker pool
- Delete [`src/streaming_recorder.py`](src/streaming_recorder.py) - functionality absorbed into transcribe.py
- Delete [`src/vad_service.py`](src/vad_service.py) - imported but never used

## Phase 2: Remove Unused Functions from transcribe.py

Delete functions that are defined but never called:

- Remove `load_templates()` (line 93) - never called
- Remove keyboard handler functions (lines 730-766):
  - `setup_keyboard_handler()`
  - `restore_keyboard_handler()`
  - `check_keyboard_input()`
  - `get_keyboard_char()`
- Remove `cleanup_services_with_timeout()` (line 856) - only referenced by VADService which is unused
- Remove VADService import and related logging config (lines 53, 87-89)

## Phase 3: Extract Settings Module

Create [`src/settings.py`](src/settings.py) and move all settings menu code (~500 lines):

- Move `update_env_setting()` helper
- Move `settings_menu()` and all submenus:
  - `transcription_settings()`
  - `output_settings()`
  - `ai_settings()`
  - `audio_settings()`
  - `advanced_performance_settings()`
  - `command_settings()`
  - `uninstall_settings()`
- Move `list_audio_devices()` helper

Update [`src/transcribe.py`](src/transcribe.py) to import: `from settings import settings_menu`

## Phase 4: Extract Commands Module

Create [`src/commands.py`](src/commands.py) and move command handlers (~400 lines):

- Move all command functions:
  - `list_transcripts()`
  - `show_transcript()`
  - `show_audio_files()`
  - `append_to_transcript()`
  - `summarize_file()`
  - `reprocess_transcript_command()`
  - `reprocess_failed_command()`
  - `list_recovery_sessions()`
  - `recover_session()`
  - `open_transcripts_folder()`
- Move helper: `transcribe_audio_file()`

Update [`src/transcribe.py`](src/transcribe.py) to import commands and wire them to CLI args.

## Phase 5: Create Summarizer Factory

Add to [`src/summarization_service.py`](src/summarization_service.py):

```python
def get_summarizer(ollama_model=None, ollama_api_url=None, 
                   ollama_timeout=None, max_content_length=None):
    """Factory function to create configured SummarizationService instance."""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    return SummarizationService(
        ollama_model=ollama_model or os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        ollama_api_url=ollama_api_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate"),
        ollama_timeout=ollama_timeout or int(os.getenv("OLLAMA_TIMEOUT", "180")),
        max_content_length=max_content_length or int(os.getenv("OLLAMA_MAX_CONTENT_LENGTH", "32000"))
    )
```

Replace all 7 `SummarizationService(...)` instantiations in [`src/transcribe.py`](src/transcribe.py) with `get_summarizer()`.

## Phase 6: Clean Up Recording Module

Simplify [`src/transcribe.py`](src/transcribe.py) to focus on core recording logic:

- Keep only: `record_audio_streaming()`, `handle_post_transcription_actions()`, `main()`
- Keep utility: `deduplicate_transcript()`, signal handler
- Remove session file helpers that are only used in recovery (move to recovery module if needed)

## Phase 7: Update Imports and Tests

- Update all internal imports to reflect new module structure
- Verify [`configure.py`](configure.py) still works with changes
- Test key workflows: record, list, append, settings menu

## Expected Result

**Before:**

- `transcribe.py`: 2061 lines (monolithic)
- 4 unused files (audio_chunker, transcription_worker, streaming_recorder, vad_service)
- 7 duplicate SummarizationService instantiations
- 200+ lines of unused functions

**After:**

- `transcribe.py`: ~400 lines (recording core only)
- `settings.py`: ~500 lines (settings menus)
- `commands.py`: ~400 lines (CLI commands)
- `summarization_service.py`: adds factory function
- Unused files deleted
- Single source of truth for summarizer configuration

### To-dos

- [ ] Delete unused source files (audio_chunker.py, transcription_worker.py, streaming_recorder.py, vad_service.py)
- [ ] Remove unused functions from transcribe.py (load_templates, keyboard handlers, cleanup_services)
- [ ] Create settings.py module and move all settings menu code from transcribe.py
- [ ] Create commands.py module and move all CLI command handlers from transcribe.py
- [ ] Add get_summarizer() factory function and replace 7 duplicate instantiations
- [ ] Simplify transcribe.py to focus on core recording logic only
- [ ] Update imports and test key workflows (record, list, append, settings)