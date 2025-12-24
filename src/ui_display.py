# src/ui_display.py
"""
Clean UI display for Rejoice transcription tool.
Provides clear, minimal status screens that replace each other.
"""

import os
import sys


def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def show_loading():
    """Show initial loading message"""
    clear_screen()
    print("Rejoice loading...")


def show_recording(duration_seconds, volume_level=0.0):
    """
    Show recording status screen

    Args:
        duration_seconds: Recording duration in seconds
        volume_level: Current volume level (0.0 to 1.0)
    """
    clear_screen()

    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)

    # Volume meter (20 bars)
    filled_bars = int(volume_level * 20)
    volume_meter = "█" * filled_bars + "░" * (20 - filled_bars)

    print("___________________________________\n")
    print("🔴 RECORDING...\n")
    print(f"⏱️  {minutes:02d}:{seconds:02d}")
    print(f"🎤 [{volume_meter}]")
    print("\nPress Enter or Ctrl+C to stop recording.")
    print("___________________________________")


def show_transcribing(elapsed_seconds, session_id=None, progress=None):
    """
    Show transcription progress screen

    Args:
        elapsed_seconds: Time elapsed in seconds
        session_id: Optional session ID
        progress: Optional progress percentage (0-100)
    """
    clear_screen()

    print("___________________________________\n")
    print("TRANSCRIBING...\n")

    if progress is not None:
        filled = int(progress * 30 / 100)
        progress_bar = "█" * filled + "░" * (30 - filled)
        print(f"PROGRESS      {progress_bar} {progress}%")

    print(f"ELAPSED       {elapsed_seconds:05.1f}s\n")

    if session_id:
        print(f"SESSION ID    {session_id}")

    print("___________________________________")


def show_processing(elapsed_seconds, session_id=None, file_name=None, progress=None):
    """
    Show processing (AI metadata) screen

    Args:
        elapsed_seconds: Time elapsed in seconds
        session_id: Optional session ID
        file_name: Optional filename
        progress: Optional progress percentage (0-100)
    """
    clear_screen()

    print("___________________________________\n")
    print("PROCESSING...\n")

    if progress is not None:
        filled = int(progress * 30 / 100)
        progress_bar = "█" * filled + "░" * (30 - filled)
        print(f"PROGRESS      {progress_bar} {progress}%")

    print(f"ELAPSED       {elapsed_seconds:05.1f}s\n")

    if session_id:
        print(f"SESSION ID    {session_id}")
    if file_name:
        print(f"FILE          {file_name}")

    print("___________________________________")


def show_complete(session_id, file_name, file_path, duration_seconds, word_count):
    """
    Show completion screen

    Args:
        session_id: Session ID
        file_name: Name of the saved file
        file_path: Full path to saved file
        duration_seconds: Total recording duration
        word_count: Number of words transcribed
    """
    clear_screen()

    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)

    print("___________________________________\n")
    print("✅ COMPLETE\n")
    print("Transcript copied to clipboard\n")
    print(f"SESSION ID    {session_id}")
    print(f"FILE          {file_name}")
    print(f"DURATION      {minutes:02d}:{seconds:02d}")
    print(f"WORDS         {word_count}\n")
    print(f"Saved to {file_path}")
    print("___________________________________\n")
