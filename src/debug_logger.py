# src/debug_logger.py

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

class DebugLogger:
    """
    Dual-output debug logger that writes detailed logs to file
    and emits concise milestones to console.
    """
    
    def __init__(self, session_id: str, save_path: str, enabled: bool = False):
        """
        Initialize debug logger.
        
        Args:
            session_id: Unique session identifier
            save_path: Directory to save debug logs
            enabled: Whether debug mode is active
        """
        self.enabled = enabled
        self.session_id = session_id
        
        if self.enabled:
            # Create debug log file
            debug_dir = Path(save_path) / "debug_logs"
            debug_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = debug_dir / f"debug_{session_id}_{timestamp}.log"
            
            # Set up file logger
            self.logger = logging.getLogger(f"debug_{session_id}")
            self.logger.setLevel(logging.DEBUG)
            
            # File handler with detailed format
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.milestone(f"Debug log started: {self.log_file}")
        else:
            self.logger = None
            self.log_file = None
    
    def milestone(self, message: str):
        """Log a high-level milestone to console."""
        if self.enabled:
            print(f"🔍 {message}")
    
    def detail(self, message: str):
        """Log detailed information to file only."""
        if self.enabled and self.logger:
            self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug-level information to file only."""
        if self.enabled and self.logger:
            self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning to both console and file."""
        if self.enabled:
            print(f"⚠️  {message}")
            if self.logger:
                self.logger.warning(message)
    
    def error(self, message: str):
        """Log error to both console and file."""
        if self.enabled:
            print(f"❌ {message}")
            if self.logger:
                self.logger.error(message)
    
    def close(self):
        """Close the debug logger."""
        if self.enabled and self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

