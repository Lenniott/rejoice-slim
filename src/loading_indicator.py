#!/usr/bin/env python3
"""
Simple loading indicator utility.
"""

import sys
import time
import threading
from typing import Optional

class LoadingIndicator:
    """
    A one-line loading indicator that can be updated and cleared.
    """
    
    def __init__(self, initial_message: str = "Processing..."):
        self.message = initial_message
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_index = 0
        self._lock = threading.Lock()
        
    def start(self, message: Optional[str] = None):
        """Start the loading indicator."""
        with self._lock:
            if self.is_running:
                return
                
            if message:
                self.message = message
                
            self.is_running = True
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()
    
    def update(self, message: str):
        """Update the loading message."""
        with self._lock:
            self.message = message
    
    def stop(self, final_message: Optional[str] = None):
        """Stop the loading indicator and optionally show final message."""
        with self._lock:
            if not self.is_running:
                return
                
            self.is_running = False
            
            if self.thread:
                self.thread.join(timeout=0.1)
            
            # Clear the current line
            print("\r" + " " * 80 + "\r", end="", flush=True)
            
            # Show final message if provided
            if final_message:
                print(final_message, flush=True)
    
    def _animate(self):
        """Animation loop for the spinner."""
        while self.is_running:
            with self._lock:
                if not self.is_running:
                    break
                    
                spinner = self.spinner_chars[self.spinner_index % len(self.spinner_chars)]
                display_text = f"\r{spinner} {self.message}"
                print(display_text, end="", flush=True)
                
                self.spinner_index += 1
            
            time.sleep(0.1)

def show_progress(message: str, duration: float = 1.0) -> LoadingIndicator:
    """
    Show a progress indicator for a specific duration.
    
    Args:
        message: Message to display
        duration: How long to show (for demo purposes)
        
    Returns:
        LoadingIndicator instance
    """
    indicator = LoadingIndicator(message)
    indicator.start()
    return indicator

# Example usage
if __name__ == "__main__":
    # Demo the loading indicator
    loader = LoadingIndicator("Processing audio...")
    loader.start()
    
    # Simulate work
    time.sleep(2)
    loader.update("Transcribing with Whisper...")
    time.sleep(2)
    loader.update("Generating summary...")
    time.sleep(1.5)
    
    loader.stop("✅ Complete!")