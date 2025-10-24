#!/bin/bash

# --- Local Transcriber Uninstall Script ---

echo "🗑️  Uninstalling Local Transcriber..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Remove aliases from ~/.zshrc
echo "🔗 Removing aliases from ~/.zshrc..."
if [ -f ~/.zshrc ]; then
    # Create a temporary file without the Local Transcriber Setup section
    awk '
    /^# Added by Local Transcriber Setup/ {
        in_section = 1
        next
    }
    /^$/ && in_section {
        in_section = 0
        next
    }
    /^# [^A]/ && in_section {
        in_section = 0
    }
    !in_section {
        print
    }
    ' ~/.zshrc > ~/.zshrc.tmp && mv ~/.zshrc.tmp ~/.zshrc
    
    echo "✅ Removed aliases from ~/.zshrc"
else
    echo "⚠️  ~/.zshrc not found, skipping alias removal"
fi

# 2. Ask about keeping transcripts
echo ""
echo "📝 Your transcripts are saved in: $PROJECT_DIR"
echo "Do you want to keep your transcripts? (y/n) [default: y]"
read -r keep_transcripts
keep_transcripts=${keep_transcripts:-y}

if [[ "$keep_transcripts" =~ ^[Yy]$ ]]; then
    echo "✅ Keeping transcripts in: $PROJECT_DIR"
else
    echo "🗑️  Removing transcripts..."
    # Get save path from .env if it exists
    if [ -f "$PROJECT_DIR/.env" ]; then
        SAVE_PATH=$(grep "^SAVE_PATH=" "$PROJECT_DIR/.env" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        if [ -n "$SAVE_PATH" ] && [ -d "$SAVE_PATH" ]; then
            echo "Removing transcripts from: $SAVE_PATH"
            rm -rf "$SAVE_PATH"
        fi
    fi
fi

# 3. Remove virtual environment
echo "🐍 Removing Python virtual environment..."
if [ -d "$PROJECT_DIR/venv" ]; then
    rm -rf "$PROJECT_DIR/venv"
    echo "✅ Removed virtual environment"
else
    echo "⚠️  Virtual environment not found"
fi

# 4. Remove configuration file
echo "⚙️  Removing configuration..."
if [ -f "$PROJECT_DIR/.env" ]; then
    rm -f "$PROJECT_DIR/.env"
    echo "✅ Removed configuration file"
else
    echo "⚠️  Configuration file not found"
fi

# 5. Final message
echo ""
echo "🎉 Uninstall complete!"
echo ""
echo "To remove this script and the entire project directory:"
echo "  rm -rf $PROJECT_DIR"
echo ""
echo "To reload your shell configuration:"
echo "  source ~/.zshrc"
echo ""
echo "Thank you for using Local Transcriber! 👋"
