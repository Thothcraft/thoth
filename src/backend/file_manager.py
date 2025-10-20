"""File management utilities for Thoth backend."""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

class FileManager:
    def __init__(self, base_path: str):
        """Initialize with the base path where data files are stored."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def list_files(self, subdir: str = '') -> List[Dict]:
        """List all files in the specified subdirectory."""
        target_dir = self.base_path / subdir if subdir else self.base_path
        if not target_dir.exists():
            return []
            
        files = []
        for item in target_dir.iterdir():
            if item.is_file():
                stat = item.stat()
                files.append({
                    'name': item.name,
                    'path': str(item.relative_to(self.base_path)),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': item.suffix[1:].lower() if item.suffix else 'unknown'
                })
            elif item.is_dir():
                files.append({
                    'name': item.name,
                    'path': str(item.relative_to(self.base_path)),
                    'type': 'directory',
                    'size': 0,
                    'created': '',
                    'modified': ''
                })
        return files

    def get_file(self, file_path: str) -> Optional[Path]:
        """Get a file path if it exists and is within the base directory."""
        try:
            full_path = (self.base_path / file_path).resolve()
            # Security check: ensure the path is within the base directory
            if self.base_path.resolve() not in full_path.parents and full_path != self.base_path.resolve():
                return None
            if full_path.exists() and full_path.is_file():
                return full_path
        except Exception:
            pass
        return None

import os
from pathlib import Path

# Use a standard data directory in the user's home directory
DATA_DIR = Path.home() / 'thoth_data'

# Initialize file manager with the data directory
file_manager = FileManager(DATA_DIR)

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
