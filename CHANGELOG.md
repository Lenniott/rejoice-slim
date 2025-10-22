# Changelog

## Version 2.1.0 - AI Hierarchical Analysis & CLI Improvements (2025-10-22)

### 🤖 Advanced AI Analysis System

**Hierarchical Summarization:**
- Intelligent chunk-based processing for large transcripts (30k+ characters)
- Breaks content into ~2000 character overlapping sections
- Each chunk analyzed for themes, questions, and actions
- Meta-summary combines all chunks into cohesive analysis
- Robust JSON parsing with automatic fallback mechanisms

**Enhanced AI Features:**
- Extracts narrative threads and main themes from conversations
- Identifies key questions asked during discussions  
- Highlights action items and decisions made
- Generates intelligent, context-aware filenames
- Creates relevant tags for categorization and search

**Improved Reliability:**
- Content truncation for optimal processing (configurable limits)
- Fallback prompts when primary analysis fails
- Enhanced JSON validation and repair
- Comprehensive error handling and logging
- Works consistently with Ollama v0.12.6+ limitations

### ⚡ Streamlined CLI Interface  

**New Short Options:**
- `rec -l` / `--list` - List all transcripts with IDs
- `rec -v ID` / `--view ID` - View transcript content  
- `rec -g ID` / `--genai ID` - AI analysis and tagging
- `rec -s` / `--settings` - Settings menu

**Renamed Commands:**
- `--summarize` → `--genai` (better describes the comprehensive AI analysis)
- `--show` → `--view` (clearer intent for content viewing)
- All original long-form commands still work for backward compatibility

**Enhanced User Experience:**
- Shorter commands for frequent operations
- Intuitive naming that reflects actual functionality  
- Complete backward compatibility maintained
- Improved help text and descriptions

### 🔧 Technical Improvements

**AI Processing:**
- Configurable content length limits (`OLLAMA_MAX_CONTENT_LENGTH`)
- Smart content sampling for fallback processing  
- JSON schema validation with detailed field checking
- Automatic quote escaping and special character handling

**System Reliability:**
- Real-time settings updates without restart required
- Enhanced error messages for debugging
- Improved file renaming with AI-generated names
- Better handling of large transcript processing

---

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