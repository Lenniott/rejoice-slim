# OS-Agnostic Roadmap for Rejoice Slim

**Status:** Mac-only (as of 2025-01-25)
**Goal:** Make Rejoice Slim fully cross-platform (macOS, Linux, Windows)

## Current State: Mac-Focused Audit

### Mac Dependencies Found

#### 1. Installation & Setup
- **Homebrew required** for PortAudio installation ([setup.sh:8](setup.sh#L8), [setup.sh:31](setup.sh#L31))
- References to macOS throughout [INSTALLATION.md](INSTALLATION.md#L44)
- Hardcoded `~/.zshrc` for shell aliases ([setup.sh:74](setup.sh#L74), [setup.sh:96](setup.sh#L96))

#### 2. File Opening Commands
- Uses `open` command (macOS-only) in multiple places:
  - [src/commands.py:29](src/commands.py#L29) - Opening folders
  - [src/transcribe.py:1010](src/transcribe.py#L1010) - Opening files
  - [src/obsidian_utils.py:188](src/obsidian_utils.py#L188) - Opening Obsidian URIs

#### 3. Example Paths in Documentation
- `/Users/` paths hardcoded in examples (USAGE.md, SETTINGS.md)
- iCloud Drive paths specific to macOS ([obsidian_utils.py:15](src/obsidian_utils.py#L15))

#### 4. Error Messages
- Platform-specific error messages ([src/transcribe.py:191](src/transcribe.py#L191), [src/settings.py:164](src/settings.py#L164))

### Already Cross-Platform ✅

- **File operations** use cross-platform checks (`sys.platform`)
- **Audio recording** uses cross-platform `sounddevice` library
- **Core transcription** uses platform-agnostic Python/Whisper
- **Most functionality** is platform-agnostic

---

## 📋 Implementation Todo List

### 🔴 Critical - Installation & Setup

- [ ] **Update setup.sh to support multiple platforms**
  - Add Linux package manager detection (apt, yum, pacman)
  - Add Windows detection and provide PowerShell alternative
  - Make Homebrew optional, provide fallback for PortAudio installation
  - Detect shell type (bash, zsh, fish, PowerShell) instead of hardcoding zshrc

- [ ] **Create platform-specific installation scripts**
  - `setup-linux.sh` - For Debian/Ubuntu, Fedora, Arch
  - `setup-windows.ps1` - PowerShell script for Windows
  - `setup-macos.sh` - Rename current setup.sh for clarity

- [ ] **Update configure.py default paths**
  - Replace hardcoded `~/Documents/transcripts` with platform-aware defaults:
    - macOS: `~/Documents/transcripts`
    - Linux: `~/Documents/transcripts` or `~/.local/share/rejoice/transcripts`
    - Windows: `%USERPROFILE%\Documents\transcripts`

### 🟡 High Priority - Cross-Platform Shell Integration

- [ ] **Abstract shell configuration in setup.sh**
  - Detect shell type: zsh, bash, fish, PowerShell
  - Support multiple shell config files: `~/.zshrc`, `~/.bashrc`, `~/.config/fish/config.fish`, PowerShell profile
  - Create helper function: `add_alias_to_shell(command, path, shell_type)`

- [ ] **Update src/settings.py command name changer**
  - Replace hardcoded `~/.zshrc` with shell detection (line 356-383)
  - Support Windows environment variables / PATH modification

### 🟠 Medium Priority - Documentation Updates

- [ ] **Update INSTALLATION.md**
  - Add Linux installation section (apt, yum, pacman)
  - Add Windows installation section (PowerShell, winget, manual)
  - Remove "macOS only" language from Homebrew section
  - Add cross-platform audio dependency instructions

- [ ] **Update README.md and USAGE.md**
  - Replace `/Users/` example paths with platform-neutral `~` or generic paths
  - Add Windows-style path examples where appropriate
  - Update "Perfect For" section to mention cross-platform support

- [ ] **Update SETTINGS.md and ARCHITECTURE.md**
  - Replace macOS-specific example paths
  - Document platform-specific behaviors

### 🟢 Low Priority - Error Messages & UX

- [ ] **Remove platform-specific error messages**
  - [src/transcribe.py:191-194](src/transcribe.py#L191-L194) - Generic Ctrl+C message
  - [src/settings.py:164-167](src/settings.py#L164-L167) - Generic cancellation message
  - [src/transcribe.py:530](src/transcribe.py#L530), [1061](src/transcribe.py#L1061), [1352](src/transcribe.py#L1352) - Standardize messages

- [ ] **Add platform detection utility module**
  - Create `src/platform_utils.py` with helpers:
    - `get_default_transcripts_path()` - Returns platform-appropriate default
    - `get_shell_config_file()` - Returns appropriate shell config
    - `get_file_opener_command()` - Returns `open`/`xdg-open`/`explorer`
    - `get_config_dir()` - Returns platform config directory

### 🔵 Enhancement - Testing & CI

- [ ] **Add platform-specific testing**
  - Create GitHub Actions workflow for Ubuntu, macOS, Windows
  - Test installation scripts on each platform
  - Test audio recording on each platform (if possible in CI)

- [ ] **Create platform compatibility matrix**
  - Document which features work on which platforms
  - Note any platform-specific limitations

### 🟣 Optional - Advanced Cross-Platform Features

- [ ] **Windows-specific enhancements**
  - Add Windows Terminal integration
  - Support Windows notification system for completion alerts
  - Test clipboard functionality on Windows

- [ ] **Linux-specific enhancements**
  - Add systemd service file for background recording
  - Support Linux notification systems (notify-send)
  - Test with various desktop environments (GNOME, KDE, XFCE)

- [ ] **Consider packaging**
  - Create PyPI package for easy `pip install rejoice-slim`
  - Create platform-specific packages (Homebrew tap, AUR, winget)

---

## Technical Analysis

### Current Cross-Platform Coverage: ~70%

**Working cross-platform:**
- Audio recording (sounddevice library)
- File I/O (Python's os/pathlib)
- Transcription engine (Whisper)
- AI processing (Ollama - if installed locally)
- Most core functionality

**Mac-only components:**
- Installation process (Homebrew, zshrc)
- Setup scripts
- Documentation examples
- Some error messages

### Priority Assessment

**Highest ROI changes:**
1. **Installation scripts** - Critical blocker for non-Mac users
2. **Documentation** - Removes confusion and sets expectations
3. **Error messages** - Polish and consistency

**Current Workaround:**
The tool CAN work on Linux/Windows today with manual setup:
- Install Python dependencies manually
- Configure .env file manually
- Set up shell aliases manually
- Ensure PortAudio is installed

But the installation experience is Mac-centric, which is the main barrier.

---

## Platform-Specific Implementation Notes

### Linux Support
- **Audio:** PortAudio via `apt install portaudio19-dev` (Debian/Ubuntu)
- **Shell:** Detect bash/zsh/fish, update appropriate rc file
- **File opening:** Use `xdg-open` (already implemented)
- **Paths:** Use `~/.local/share/rejoice/` for config (XDG standard)

### Windows Support
- **Audio:** PortAudio via pip should work, or use Windows audio API
- **Shell:** PowerShell profile or CMD batch files
- **File opening:** Use `start` or `explorer` (already implemented)
- **Paths:** Use `%APPDATA%\Rejoice\` or `%USERPROFILE%\Documents\Transcripts\`
- **Challenges:** Different path separators, permissions model

### Implementation Strategy

**Phase 1: Installation (Weeks 1-2)**
- Create platform detection in setup scripts
- Add Linux package manager support
- Create Windows PowerShell installer

**Phase 2: Documentation (Week 3)**
- Update all docs with platform-neutral language
- Add platform-specific installation instructions
- Create troubleshooting guides per platform

**Phase 3: Testing (Week 4)**
- Set up CI/CD for all platforms
- Test on real hardware
- Create platform compatibility matrix

**Phase 4: Polish (Week 5+)**
- Platform-specific UX improvements
- Package management (PyPI, Homebrew, AUR)
- Community feedback and iteration

---

## Conclusion

Rejoice Slim is architecturally sound for cross-platform support. The core functionality is already platform-agnostic thanks to good library choices (sounddevice, Python standard library).

**The main barrier is the installation experience**, which is currently Mac-only. Addressing this would unlock the tool for Linux and Windows users with relatively modest engineering effort.

**Estimated effort:** 2-4 weeks of focused development for full cross-platform support, with most time spent on installation scripts and testing rather than core functionality changes.
