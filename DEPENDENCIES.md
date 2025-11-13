# 📦 Dependencies Guide

**← [Back to Home](README.md)**

## 🔍 Package Overview

Rejoice uses carefully selected packages for privacy, performance, and reliability. All processing happens locally on your device.

## 🐍 Core Python Dependencies

### 🎤 Audio Processing
```python
sounddevice>=0.4.6
# Real-time audio recording from microphone
# Why: Cross-platform, low-latency audio capture
# Privacy: Local only, no network access
# Works: Concurrent recording (Zoom, music, etc.)
```

```python
numpy>=1.24.0
# Numerical operations for audio data processing
# Why: Efficient audio buffer manipulation, required by sounddevice & Whisper
# Privacy: Mathematical library, no data collection
# Size: ~20MB, fundamental to audio processing
```

```python
scipy>=1.11.0
# Scientific computing library for audio operations
# Why: Audio file format support (WAV, MP3, etc.)
# Usage: Required by sounddevice for sample rate conversion
# Privacy: Mathematical library, no data collection
# Size: ~50MB, handles audio format conversions
```

### 🤖 AI Transcription
```python
openai-whisper>=20231117
# Local speech-to-text AI model (OpenAI's Whisper)
# Why: Best accuracy for offline transcription, runs locally
# Privacy: NO API calls to OpenAI - completely offline after download
# Models: Downloads 39MB-1.5GB models to your computer
# Security: Open source, auditable, runs in isolated environment
```

```python
torch>=2.0.0
# PyTorch machine learning framework (required by Whisper)
# Why: Whisper models run on PyTorch for AI inference
# Privacy: Local computation only, no telemetry
# Installation: Auto-installed as Whisper dependency (not in requirements.txt)
# Size: ~800MB-2GB depending on your system (CPU vs GPU)
# Note: Large but essential for AI model execution
```

### 🗂️ File Handling
```python
python-dotenv>=1.0.0
# Environment variable management for configuration
# Why: Secure storage of user settings in .env files
# Privacy: Local file storage, no external access
# Size: <1MB, lightweight configuration management
```

```python
PyYAML>=6.0
# YAML parsing for transcript file headers
# Why: Creates structured frontmatter with ID, title, creation date, status
# Privacy: Local file processing only
# Usage: Enables ID-based transcript system with metadata
# Size: <1MB, lightweight text processing
# Alternative: JSON (but YAML is more human-readable for frontmatter)
```

### 🌐 Local AI Integration (Optional)
```python
requests>=2.25.1
# HTTP client for Ollama API communication
# Why: Communicate with local Ollama instance (localhost:11434)
# Usage: AI-generated filenames and summaries
# Privacy: ONLY connects to localhost - no external requests
# Security: All traffic stays on your computer
# Without Ollama: This dependency is unused, no network activity
# Size: <1MB, lightweight HTTP client
```

### 🖥️ System Integration
```python
pyperclip>=1.8.2
# Clipboard integration for auto-copy feature
# Why: Automatically copy transcripts to clipboard after recording
# Privacy: Local system clipboard only, no network access
# Platform: Cross-platform clipboard support
# Optional: Can be disabled via AUTO_COPY=false
# Size: <100KB, pure system integration
```

## 🖥️ System Dependencies

### macOS Requirements

#### PortAudio (via Homebrew)
```bash
brew install portaudio
```
- **What it is:** Professional audio I/O library
- **Why needed:** Reliable microphone access while other apps are running
- **Security:** Open source, widely used, maintained by audio professionals
- **Alternatives:** Basic audio (but may have conflicts with Zoom, Spotify, etc.)

#### Xcode Command Line Tools
```bash
xcode-select --install
```
- **What it is:** System compilers and build tools
- **Why needed:** Compile Python packages with C extensions
- **Security:** Official Apple development tools
- **Size:** ~500MB, one-time installation

### Linux Requirements

#### System Audio Libraries
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# Fedora/CentOS
sudo dnf install portaudio-devel python3-pyaudio

# Arch Linux
sudo pacman -S portaudio python-pyaudio
```

### Windows Requirements

#### Visual Studio Build Tools
- **Download:** [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **Why needed:** Compile Python packages with C extensions
- **Alternative:** Full Visual Studio Community (free)

## 🤖 AI Models

### Whisper Models (Required)
Downloaded automatically during setup:

| Model  | Size    | RAM Usage | Accuracy | Use Case |
|--------|---------|-----------|----------|----------|
| tiny   | 39 MB   | ~390 MB   | Good     | Testing, low-resource systems |
| base   | 74 MB   | ~500 MB   | Better   | General use, older computers |
| small  | 244 MB  | ~1 GB     | Very Good| **Recommended default** |
| medium | 769 MB  | ~2 GB     | Excellent| Professional transcription |
| large  | 1550 MB | ~4 GB     | Best     | Critical accuracy needs |

**Storage location:** `~/.cache/whisper/` (can be deleted to free space)

### Ollama Models (Optional)
For AI-generated filenames and metadata:

| Model      | Size  | RAM Usage | Quality | Download Command |
|------------|-------|-----------|---------|------------------|
| gemma3:4b  | 2.5GB | ~3 GB     | Excellent | `ollama pull gemma3:4b` |
| llama3     | 4.7GB | ~5 GB     | Excellent | `ollama pull llama3` |
| qwen3:0.6b | 350MB | ~500 MB   | Good    | `ollama pull qwen3:0.6b` |
| phi3       | 2.3GB | ~3 GB     | Very Good | `ollama pull phi3` |

**Storage location:** `~/.ollama/models/` (managed by Ollama)

## 🔒 Security Information

### Privacy Guarantees
- ✅ **All audio processing on your device** - Whisper runs locally
- ✅ **No OpenAI API calls** - Uses downloaded models, not cloud services
- ✅ **No cloud AI services** - Ollama runs locally
- ✅ **No internet required** after initial setup
- ✅ **Zero telemetry** - No usage data collected
- ✅ **Zero external requests** (except to localhost Ollama)

### Network Activity Verification
The only network requests are to `localhost:11434` (local Ollama, if enabled):
```python
# From src/summarization_service.py
response = requests.get("http://localhost:11434/api/version", ...)  # Check if Ollama is running
response = requests.post("http://localhost:11434/api/generate", ...)  # Generate summaries/tags
```

You can verify this by:
1. **Reading the source code** - All network requests are visible and localhost-only
2. **Network monitoring** - Use tools like Little Snitch, Wireshark
3. **Offline testing** - Disconnect internet, transcription still works (AI features require Ollama)

### Package Security
- **All packages from PyPI** - Official Python package repository
- **Open source dependencies** - Code is auditable
- **No proprietary packages** - Everything can be inspected
- **Minimal dependencies** - Reduced attack surface

## 🗑️ Disk Space Usage

### Typical Installation
- **Python packages:** ~100MB
- **Whisper small model:** ~244MB  
- **Ollama + model:** ~2.5GB (optional)
- **System dependencies:** ~500MB
- **Total:** ~1GB (without Ollama) or ~3.5GB (with Ollama)

### Cleaning Up
```bash
# Remove Whisper model cache
rm -rf ~/.cache/whisper/

# Remove specific Ollama model  
ollama rm gemma3:4b

# Uninstall Python packages
pip uninstall -r requirements.txt

# Remove Homebrew audio (macOS)
brew uninstall portaudio
```

## 🚀 Performance Optimization

### For Lower RAM Usage
1. **Use smaller Whisper model:** `tiny` or `base`
2. **Use smaller Ollama model:** `qwen3:0.6b` 
3. **Reduce chunk duration:** Less buffering
4. **Fewer worker threads:** Lower parallel processing

### For Better Accuracy
1. **Use larger Whisper model:** `medium` or `large`
2. **More RAM:** Enables larger models
3. **Better microphone:** Hardware quality matters
4. **Quiet environment:** Reduce background noise

### For Faster Processing
1. **SSD storage:** Faster model loading
2. **More CPU cores:** Better parallel processing
3. **GPU acceleration:** PyTorch can use CUDA/Metal
4. **Optimal chunk size:** Balance latency vs efficiency

## ❓ Dependency FAQ

### "Why not use OpenAI's API directly?"
- **Privacy:** API sends your audio to their servers
- **Cost:** API charges per minute of audio
- **Offline:** API requires internet connection
- **Control:** Local models give you full control

### "Can I use different AI models?"
- **Whisper:** Stick with official OpenAI models (best accuracy)
- **Ollama:** Try different models for metadata generation
- **Custom:** Advanced users can modify the code

### "Why so many dependencies?"
- **Audio:** Cross-platform audio is complex
- **AI:** Modern AI models require substantial frameworks
- **Reliability:** Fallbacks and error handling need extra packages
- **Features:** Rich functionality requires specialized libraries

### "Is this bloated?"
- **Compared to cloud services:** Larger initial download, but then completely offline
- **Compared to professional tools:** Much smaller than Adobe Audition, etc.
- **Functionality:** Each package serves a specific, essential purpose

## 🔧 Alternative Configurations

### Minimal Setup (Transcription only)
```bash
pip install openai-whisper sounddevice scipy numpy python-dotenv
# ~300MB total, basic transcription with timestamp filenames
```

### Full Features (Without Ollama)
```bash
pip install -r requirements.txt
# All features except AI-generated filenames/summaries
# Uses ID_DATE_transcript.md format for files
```

### AI-Enhanced Setup (Recommended)
```bash
pip install -r requirements.txt
brew install ollama  # or download from ollama.ai
ollama pull gemma3:4b
# Full features with smart filenames and summaries
```

---

**← [Back to Home](README.md)** | **Previous: [Settings Guide ←](SETTINGS.md)**