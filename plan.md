I'll analyze the current codebase and create a comprehensive plan for implementing parallel streaming transcription while cleaning up obsolete code. 

Read [](file:///Users/benjamin.mizrany/Repos/rejoice-slim/src/transcribe.py#842-842), lines 842 to 942

Read [](file:///Users/benjamin.mizrany/Repos/rejoice-slim/src/transcription_worker.py)

Read [](file:///Users/benjamin.mizrany/Repos/rejoice-slim/src/audio_chunker.py)

Created 8 todos

# Parallel Streaming Transcription Implementation Plan

## Overview
Transform the current three-tier fallback system into a true parallel streaming transcription system where Whisper processes audio chunks simultaneously with recording, eliminating the wait time at the end.

## Architecture Changes

### Current Problems to Solve:
1. **Complex three-tier fallback** (real-time → chunk recovery → full transcription)
2. **End-of-recording wait time** for transcription processing
3. **Duplicate chunk management systems** (completed_chunks, failed_chunks, chunk_order)
4. **Overly complex error recovery** that often just falls back to full transcription anyway

### New Approach:
1. **Single streaming pipeline** with parallel processing
2. **Live transcription assembly** as chunks complete
3. **Minimal fallback** to full transcription only if streaming completely fails
4. **Real-time user feedback** showing transcription as it happens

## Detailed Implementation Plan 

Starting (1/8) *Analyze current code architecture*

### Step 1: Analysis of Current Code (In Progress)

Looking at the current `record_audio_chunked()` function, I can see several components that add complexity without significant benefit:

**Components to Remove:**
- `completed_chunks = {}` - Complex chunk state tracking
- `failed_chunks = {}` - Retry logic that rarely succeeds
- `chunk_order = []` - Order management (timestamp-based sorting is simpler)
- Three-tier fallback strategy (real-time → chunk recovery → full transcription)
- Complex chunk recovery and reassembly logic

**Components to Simplify:**
- Worker pool management (reduce from complex error handling to simple parallel processing)
- VAD service integration (optional, not critical for core functionality)
- Keyboard handling (keep but simplify)

**Components to Keep:**
- Session file management for recovery
- Audio writer for continuous disk saving
- Signal handling for graceful shutdown

### Step 2: TranscriptionWorkerPool Modifications

The current worker pool needs these additions:

```python
# New methods to add to TranscriptionWorkerPool:

def add_chunk_with_id(self, chunk_id: str, audio_data: np.ndarray, timestamp: float):
    """Add chunk with ID and timestamp for tracking"""
    
def get_completed_results(self, timeout: float = 0.5) -> List[Tuple[str, str, float]]:
    """Get all completed transcription results"""
    
def get_streaming_transcript(self) -> str:
    """Get current assembled transcript from completed chunks"""
```

### Step 3: New Streaming Function Architecture

The new `record_audio_chunked()` will have this simplified flow:

1. **Setup Phase** (same as current)
   - Initialize audio writer
   - Load Whisper model
   - Start worker pool and chunker

2. **Streaming Phase** (completely new)
   - Audio callback writes to disk AND sends to chunker
   - Chunker provides ready chunks to worker pool immediately
   - Background thread continuously assembles completed results
   - Live display shows transcription as it appears

3. **Assembly Phase** (drastically simplified)
   - Stop recording
   - Wait 1-2 seconds for final chunks to complete
   - Sort results by timestamp and join
   - Minimal deduplication
   - Return result immediately

4. **Fallback Phase** (only if streaming fails completely)
   - Full file transcription as last resort

### Step 4: Components to Remove

**Functions to completely remove:**
- Complex chunk tracking in `record_audio_chunked()`
- Chunk recovery mechanisms
- Multi-tier fallback logic in the main recording loop

**Simplifications:**
- Reduce worker pool retry attempts (2 max instead of 4)
- Remove complex VAD integration (optional feature)
- Simplify error handling (fail fast, don't retry extensively)

### Step 5: Performance Optimizations

**New configuration defaults:**
```python
# Optimized for streaming performance
CHUNK_DURATION_SECONDS = 6.0  # Smaller chunks for faster response
TRANSCRIPTION_WORKER_THREADS = 2  # Parallel processing
STREAMING_DISPLAY_INTERVAL = 1.5  # Update live display every 1.5s
```

### Step 6: Clean Up

**Remove these imports** (if no longer used):
```python
# Complex threading that will be simplified
# Excessive retry/recovery imports
# VAD-specific imports if not used in streaming mode
```

**Simplify configuration:**
- Remove chunk recovery settings
- Remove complex retry configurations
- Focus on core streaming parameters

## Implementation Benefits

1. **Faster Results**: Transcription available immediately when recording stops
2. **Live Feedback**: Users see transcription happening in real-time
3. **Simpler Code**: Remove ~200 lines of complex fallback logic
4. **Better Performance**: Parallel processing with optimized chunk sizes
5. **Fault Tolerance**: Still has full-file fallback if streaming fails completely

## Backward Compatibility

- All existing CLI commands work the same
- Session recovery still available for failed recordings
- High-accuracy mode can disable streaming for maximum quality
- Environment variables remain the same

