# src/obsidian_utils.py

import os
import sys
import urllib.parse
import subprocess
from typing import List, Tuple, Optional


def select_vault_from_path(save_path: str) -> Optional[str]:
    """
    Parse path into components and let user select vault directory.

    Example:
        Input: /Users/benjamin/Library/Mobile Documents/.../Gilgamesh_house/30-39 Resources/30 Transcripts
        Shows:
            1. Users
            2. benjamin
            3. Library
            4. Mobile Documents
            5. iCloud~md~obsidian
            6. Documents
            7. Gilgamesh_house
            8. 30-39 Resources
            9. 30 Transcripts

        User enters: 7
        Returns: /Users/benjamin/Library/Mobile Documents/iCloud~md~obsidian/Documents/Gilgamesh_house

    Args:
        save_path: The full save path to parse

    Returns:
        Full path to selected directory, or None if cancelled
    """
    # Normalize and expand the path
    abs_path = os.path.abspath(os.path.expanduser(save_path))

    # Split into components and build cumulative paths
    parts = []
    cumulative_paths = []
    current = abs_path

    # Build list from bottom up, then reverse
    while current != os.path.dirname(current):  # Stop at root
        parent = os.path.dirname(current)
        part = os.path.basename(current)
        if part:  # Skip empty parts
            parts.insert(0, part)
            cumulative_paths.insert(0, current)
        current = parent

    if not parts:
        print("Error: Could not parse path into components")
        return None

    # Display numbered list
    print("\nDetected directories from path:")
    for i, part in enumerate(parts, 1):
        print(f"  {i}. {part}")

    # Get user selection
    while True:
        try:
            choice = input(f"\nWhich directory is your Obsidian vault? [1-{len(parts)}, or 0 to cancel]: ").strip()

            if choice == '0':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(parts):
                selected_path = cumulative_paths[idx]
                selected_name = parts[idx]
                print(f"\n✓ Selected vault: {selected_name}")
                print(f"  Full path: {selected_path}")
                return selected_path
            else:
                print(f"Please enter a number between 1 and {len(parts)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled")
            return None


def is_path_in_vault(file_path: str, vault_path: str) -> bool:
    """
    Check if a file path is within a vault.

    Args:
        file_path: Path to check
        vault_path: Vault root path

    Returns:
        True if file_path is within vault_path
    """
    file_abs = os.path.abspath(os.path.expanduser(file_path))
    vault_abs = os.path.abspath(os.path.expanduser(vault_path))

    # Check if file_abs starts with vault_abs
    return file_abs.startswith(vault_abs + os.sep) or file_abs == vault_abs


def build_obsidian_uri(file_path: str, vault_path: str) -> str:
    """
    Build Obsidian URI for opening a file.

    Args:
        file_path: Absolute path to the file
        vault_path: Absolute path to the vault root

    Returns:
        Obsidian URI string (obsidian://open?vault=...&file=...)
    """
    vault_name = os.path.basename(vault_path)
    abs_file_path = os.path.abspath(file_path)

    # Get relative path from vault root
    rel_path = os.path.relpath(abs_file_path, vault_path)

    # Convert to forward slashes (Obsidian expects this)
    rel_path = rel_path.replace(os.sep, "/")

    # URL-encode vault and file
    vault_enc = urllib.parse.quote(vault_name, safe='')
    file_enc = urllib.parse.quote(rel_path, safe='')

    return f"obsidian://open?vault={vault_enc}&file={file_enc}"


def configure_obsidian_integration(save_path: str) -> Tuple[bool, str]:
    """
    Configure Obsidian integration by asking user to enable and select vault.

    Args:
        save_path: The save path that will be used for transcripts

    Returns:
        Tuple of (enabled: bool, vault_path: str)
    """
    print("\n" + "="*70)
    print("📝 OBSIDIAN INTEGRATION")
    print("="*70)

    # Step 1: Ask if user wants Obsidian integration
    while True:
        choice = input("\nIntegrate with Obsidian? [y/n]: ").strip().lower()
        if choice in ['y', 'yes']:
            break
        elif choice in ['n', 'no']:
            print("\n✓ Obsidian integration disabled")
            return False, ""
        else:
            print("Please enter 'y' or 'n'")

    # Step 2: User already has save_path, use it to select vault
    print(f"\nUsing save path: {save_path}")

    # Step 3: Parse path and let user select vault
    vault_path = select_vault_from_path(save_path)

    if vault_path:
        print("\n✓ Obsidian integration configured successfully")
        return True, vault_path
    else:
        print("\n✗ Obsidian integration cancelled")
        return False, ""


def open_in_obsidian(file_path: str, vault_path: str) -> bool:
    """
    Open a file in Obsidian using URI scheme.

    Args:
        file_path: Path to the file to open
        vault_path: Path to the vault root

    Returns:
        True if successful, False otherwise
    """
    if not is_path_in_vault(file_path, vault_path):
        return False

    uri = build_obsidian_uri(file_path, vault_path)

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", uri], check=True, timeout=2)
        elif sys.platform == "win32":
            subprocess.run(["cmd", "/c", "start", "", uri], check=True, timeout=2)
        else:
            subprocess.run(["xdg-open", uri], check=True, timeout=2)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
