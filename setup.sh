#!/bin/bash

# --- Local Transcriber Setup Script ---

echo "🚀 Starting the setup for your local transcriber..."

# Check for Homebrew on macOS to install dependencies if needed
if [[ "$(uname)" == "Darwin" ]] && ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install it from https://brew.sh before running this script again."
    exit 1
fi

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please download and install it from https://www.python.org/downloads/ and then run this script again."
    exit 1
fi

# 1.5. Check for Ollama (optional)
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed. AI features (smart filenames, summaries, tags) will be disabled."
    echo "   You can install it later from https://ollama.ai to enable these features."
    echo "   The transcriber will still work perfectly for basic transcription!"
else
    echo "✅ Ollama detected - AI features will be available."
fi

# 2. Check for PortAudio (dependency for sounddevice)
if [[ "$(uname)" == "Darwin" ]] && ! brew list portaudio &> /dev/null; then
    echo "PortAudio not found. Installing via Homebrew..."
    brew install portaudio
elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]] && ! dpkg -s portaudio19-dev &> /dev/null; then
    echo "PortAudio not found. Installing via apt-get..."
    sudo apt-get update && sudo apt-get install -y portaudio19-dev
fi

# 3. Create Python Virtual Environment
echo "📦 Creating a Python virtual environment in './venv'..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment."
    exit 1
fi

# 4. Activate Virtual Environment and Install Dependencies
source venv/bin/activate
echo "🐍 Installing Python packages from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python packages."
    exit 1
fi

# 5. Run the interactive Python configuration script
echo "⚙️ Now running the interactive configuration..."
python3 configure.py

# 6. Clean up old aliases and create new ones
echo "🔗 Setting up aliases in your ~/.zshrc file..."
PROJECT_DIR="$(pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

# Get the command name from .env file 
if [ -f ".env" ]; then
    # Extract command name from COMMAND_NAME variable
    COMMAND_NAME=$(grep "^COMMAND_NAME=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    
    if [ -n "$COMMAND_NAME" ]; then
        # Remove any existing Local Transcriber Setup section from .zshrc
        if [ -f ~/.zshrc ]; then
            # Create a temporary file without the old setup section
            awk '
            /^# Added by Local Transcriber Setup/ {
                in_section = 1
                next
            }
            /^# [^A]/ && in_section {
                in_section = 0
            }
            /^$/ && in_section {
                in_section = 0
            }
            !in_section {
                print
            }
            ' ~/.zshrc > ~/.zshrc.tmp && mv ~/.zshrc.tmp ~/.zshrc
        fi
        
        # Add fresh aliases
        echo -e "\n# Added by Local Transcriber Setup" >> ~/.zshrc
        echo "alias $COMMAND_NAME='$VENV_PYTHON $PROJECT_DIR/src/transcribe.py'" >> ~/.zshrc
        echo "alias $COMMAND_NAME-settings='$VENV_PYTHON $PROJECT_DIR/src/transcribe.py --settings'" >> ~/.zshrc
        
        # Create the open-transcripts alias as well
        SAVE_PATH=$(grep "^SAVE_PATH=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        echo "alias open-transcripts='open \"$SAVE_PATH\"'" >> ~/.zshrc
        
        echo "✅ Created aliases '$COMMAND_NAME', '$COMMAND_NAME-settings' and 'open-transcripts' that use the virtual environment"
    else
        echo "❌ Could not find COMMAND_NAME in .env file"
        exit 1
    fi
else
    echo "❌ .env file not found. Configuration may have failed."
    exit 1
fi

echo ""
echo "✅ All done! Please restart your terminal or run 'source ~/.zshrc' to start using your new commands."