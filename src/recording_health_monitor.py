# src/recording_health_monitor.py

import time
from typing import Dict, List


class RecordingHealthMonitor:
    """Real-time monitoring of recording health to detect failures early"""

    def __init__(self):
        self.last_callback_time = time.time()
        self.total_frames_written = 0
        self.total_bytes_written = 0
        self.callback_failures = 0
        self.consecutive_empty_callbacks = 0
        self.write_failures = 0

        # Thresholds for health alerts
        self.CALLBACK_TIMEOUT_SECONDS = 5.0  # Alert if no callback for 5s
        self.MAX_CONSECUTIVE_EMPTY = 50  # Alert if 50 consecutive empty callbacks
        self.MAX_WRITE_FAILURES = 10  # Alert if 10 write failures

    def track_callback(self, frames_count: int, bytes_written: int, write_success: bool):
        """
        Called from audio_callback on every invocation to track recording health.

        Args:
            frames_count: Number of frames in this callback
            bytes_written: Number of bytes successfully written to file
            write_success: Whether the write operation succeeded
        """
        self.last_callback_time = time.time()
        self.total_frames_written += frames_count

        if write_success and bytes_written > 0:
            self.total_bytes_written += bytes_written
            self.consecutive_empty_callbacks = 0
        else:
            self.write_failures += 1
            if frames_count == 0:
                self.consecutive_empty_callbacks += 1

    def check_health(self) -> Dict:
        """
        Check current health status and return any issues detected.

        Returns:
            dict: Health status with keys:
                - is_healthy: bool, True if no issues
                - is_critical: bool, True if recording should stop
                - issues: List[str], list of issue descriptions
                - bytes_written: int, total bytes written
                - time_since_callback: float, seconds since last callback
        """
        time_since_callback = time.time() - self.last_callback_time

        issues = []
        is_critical = False

        # Check 1: Callback timeout (device dropout)
        if time_since_callback > self.CALLBACK_TIMEOUT_SECONDS:
            issues.append(f"No audio data for {time_since_callback:.1f}s - possible device dropout")
            is_critical = True

        # Check 2: Write failures
        if self.write_failures > self.MAX_WRITE_FAILURES:
            issues.append(f"{self.write_failures} write failures - disk or permission issue")
            is_critical = True

        # Check 3: Consecutive empty callbacks
        if self.consecutive_empty_callbacks > self.MAX_CONSECUTIVE_EMPTY:
            issues.append(f"{self.consecutive_empty_callbacks} consecutive empty callbacks")
            # Not critical - might just be silence

        return {
            'is_healthy': len(issues) == 0,
            'is_critical': is_critical,
            'issues': issues,
            'bytes_written': self.total_bytes_written,
            'time_since_callback': time_since_callback
        }

    def get_stats(self) -> Dict:
        """
        Get current statistics for display.

        Returns:
            dict: Statistics with keys:
                - total_bytes: int, total bytes written
                - total_mb: float, total megabytes written
                - last_callback: float, seconds since last callback
        """
        return {
            'total_bytes': self.total_bytes_written,
            'total_mb': self.total_bytes_written / 1024 / 1024,
            'last_callback': time.time() - self.last_callback_time
        }

    def reset(self):
        """Reset all counters for a new recording session"""
        self.last_callback_time = time.time()
        self.total_frames_written = 0
        self.total_bytes_written = 0
        self.callback_failures = 0
        self.consecutive_empty_callbacks = 0
        self.write_failures = 0
