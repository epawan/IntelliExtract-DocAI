"""
Utility functions for Document AI.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_config_path() -> Path:
    """Get the config directory path."""
    return get_project_root() / 'config'


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """Save data to JSON file."""
    output_dir = os.path.dirname(filepath)
    if output_dir:  # Only create if directory path is not empty
        os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def list_documents(directory: str, extensions: List[str] = None) -> List[str]:
    """
    List all document files in a directory.
    
    Args:
        directory: Path to directory
        extensions: List of file extensions to include
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
    
    directory = Path(directory)
    files = []
    
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in files])


def extract_doc_id(filepath: str) -> str:
    """Extract document ID from filepath."""
    return Path(filepath).stem


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union
