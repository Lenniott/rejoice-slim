# Lossless Audio Recording System

## Overview
Your transcription system now has **guaranteed lossless recording** - audio is never lost, even if the system crashes, freezes, or is interrupted.

## How It Works

### During Recording
- Audio is **immediately written to disk** as it's captured
- Uses `fsync()` to force the OS to write data to physical disk
- Session files stored in `audio_sessions/` directory
- Each session has unique timestamp-based ID

### On Success
- Complete audio file is transcribed
- Transcript is saved normally
- Session file is **automatically deleted**
- No cleanup needed from user

### On Failure/Crash/Interruption
- Session file is **automatically preserved**
- User is informed about recovery options
- Session can be recovered anytime later

## Recovery Commands

```bash
# List all recoverable sessions
python transcribe.py --list-sessions

# Recover the latest session
python transcribe.py --recover latest

# Recover specific session by ID
python transcribe.py --recover 1730745123
```

## Session Information
When sessions are preserved, you'll see:
```
💾 Audio session preserved: session_1730745123.wav
📊 Duration: 45.3s, Size: 2.1MB
🔧 Reason: cancelled
🔄 Recover with: python transcribe.py --recover 1730745123
```

## Key Benefits

✅ **Zero Audio Loss** - Even system crashes won't lose your recording
✅ **Automatic Management** - Success = cleanup, failure = preserve  
✅ **Easy Recovery** - Simple commands to process interrupted recordings
✅ **Session Metadata** - Duration, size, timestamp for each session
✅ **Immediate Safety** - Audio written to disk continuously, not buffered

## Technical Details

- Audio written as 16-bit PCM WAV files
- Real-time buffer kept small (30 seconds max)
- Session files use timestamp IDs for uniqueness
- Recovery transcribes complete saved audio file
- Cleanup happens only after successful transcription

## File Locations

- Session files: `{SAVE_PATH}/audio_sessions/session_{timestamp}.wav`
- Successful transcripts: Normal transcript location
- Recovered transcripts: Saved as "recovered_recording" + timestamp

This system ensures your audio is **never lost** while maintaining the existing real-time transcription features.