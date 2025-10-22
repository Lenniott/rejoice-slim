# Changelog

## Version 2.0.0 - ID-Based Transcript System (2025-10-22)

### 🆔 Major New Feature: ID-Based Transcript Management

**New File Naming System:**
- Transcripts use smart naming: `descriptive-name_DDMMYYYY_000001.md`
- AI-generated descriptive names (or fallback to "transcript")
- Date stamp (DDMMYYYY format) for easy chronological sorting
- 6-digit sequential ID for unique referencing
- Easy to reference by ID while maintaining descriptive filenames

**New Commands:**
- `rec --list` - List all transcripts with their IDs and creation dates
- `rec --show 000001` - Display content of transcript by ID
- `rec -000001` - Record new audio and append to existing transcript
- Backward compatible with all existing timestamp-based files

**File Structure Changes:**
- New YAML frontmatter with `id`, `title`, `created`, and `status` fields
- Title defaults to the same as ID (simplified approach)
- Clean, minimal header format for better organization

**Benefits:**
- **Easy referencing**: Use `rec -000042` instead of long filenames
- **Append functionality**: Add to existing transcripts seamlessly
- **Better organization**: Sequential IDs are easier to manage
- **Future-ready**: Foundation for advanced features like search and tagging
- **Backward compatible**: Existing files remain fully accessible

### Technical Implementation:
- Added `TranscriptIDGenerator` for unique ID generation
- Added `TranscriptHeader` for YAML frontmatter management  
- Added `TranscriptFileManager` for file operations
- Updated main transcription workflow to use ID system
- Comprehensive test coverage for all new functionality

### Migration:
- **No action required**: Existing transcripts remain unchanged
- **Gradual transition**: New recordings automatically use ID system
- **Dual support**: List command shows both new and legacy formats
- **Full compatibility**: Edit and reference old files as before

---

## Previous Versions

See git history for previous changes.