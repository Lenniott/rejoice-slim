# Real-Time Audio Chunking Architecture

## Overview

The Rejoice voice transcription tool has been enhanced with real-time audio chunking capabilities. This allows for near real-time transcription processing without waiting for the entire recording to complete.

## Architecture Components

### 1. AudioChunker (`src/audio_chunker.py`)
- **Purpose**: Segments continuous audio stream into overlapping chunks
- **Key Features**:
  - Configurable chunk duration and overlap
  - Rolling buffer to prevent data loss at boundaries
  - Thread-safe operations
- **Configuration**:
  - `CHUNK_DURATION_SECONDS`: Duration of each chunk (default: 10s)
  - `CHUNK_OVERLAP_SECONDS`: Overlap between chunks (default: 1s)

### 2. TranscriptionWorkerPool (`src/transcription_worker.py`)
- **Purpose**: Processes audio chunks concurrently using multiple worker threads
- **Key Features**:
  - Retry logic with exponential backoff
  - Error handling and recovery
  - Thread-safe result collection
- **Configuration**:
  - `TRANSCRIPTION_WORKER_THREADS`: Number of worker threads (default: 2)
  - `MAX_RETRY_ATTEMPTS`: Retry attempts for failed chunks (default: 3)

### 3. VADService (`src/vad_service.py`)
- **Purpose**: Voice Activity Detection for future auto-stop feature
- **Key Features**:
  - Energy-based voice detection (placeholder implementation)
  - Silence duration tracking
  - Foundation for automatic recording termination
- **Configuration**:
  - `SILENCE_TRIGGER_CHUNKS`: Chunks of silence before auto-stop (default: 30)

## Data Flow

```
Microphone → sounddevice callback → AudioChunker (rolling buffer)
    ↓
Queue (thread-safe)
    ↓
Worker Pool (concurrent) → Whisper API → transcribed_segments (list)
    ↓ (also)
VADService (placeholder) → silence counter (future feature)
```

## Key Benefits

1. **Real-Time Processing**: Audio is processed as it's captured, not after recording stops
2. **No Data Loss**: Overlapping chunks ensure words at boundaries are captured
3. **Fault Tolerance**: Failed chunks are retried and logged, don't crash the system
4. **Scalable**: Multiple worker threads can process chunks concurrently
5. **Memory Efficient**: All processing happens in RAM, no temporary files

## Configuration

All chunking parameters are configurable via the `.env` file:

```bash
# Real-Time Chunking Settings
CHUNK_DURATION_SECONDS=10
CHUNK_OVERLAP_SECONDS=1
TRANSCRIPTION_WORKER_THREADS=2
MAX_RETRY_ATTEMPTS=3

# VAD Settings (for future auto-stop feature)
SILENCE_TRIGGER_CHUNKS=30
```

## Error Handling

- **Chunk Processing Failure**: Retry 3 times with exponential backoff, then insert placeholder
- **Critical Failures**: Graceful shutdown with partial transcript preservation
- **Worker Thread Crashes**: Isolated failures don't affect other workers

## Performance Characteristics

- **Memory Usage**: ~100KB per 10-second chunk at 16kHz
- **Latency**: Near real-time (limited by Whisper processing time)
- **Throughput**: Scales with number of worker threads
- **Reliability**: Fault-tolerant with comprehensive error handling

## Future Enhancements

The VAD service provides the foundation for:
- Automatic recording termination on prolonged silence
- Real-time speech detection indicators
- Advanced audio analysis features
