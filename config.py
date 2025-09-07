"""Configuration for ASR evaluation project."""

import os
from pathlib import Path
from typing import List, Dict, Any

# Data paths - external data location
DATA_ROOT = Path("/homes/hinge/Projects/data-samples/data")
SYNTHETIC_JOURNALS_PATH = DATA_ROOT / "synthetic_journals"

# Supported audio formats
AUDIO_EXTENSIONS = ['.m4a', '.wav', '.mp3', '.flac']

# Output paths - local to this project
RESULTS_DIR = Path("results")
MODELS_OUTPUT_DIR = RESULTS_DIR / "models"
EVALUATION_OUTPUT_DIR = RESULTS_DIR / "evaluations"

def get_dataset_files() -> List[Dict[str, str]]:
    """
    Get list of audio-transcription pairs from the external data directory.
    
    Returns:
        List of dicts with 'audio_file' and 'transcription_file' keys
    """
    dataset_files = []
    
    if not SYNTHETIC_JOURNALS_PATH.exists():
        print(f"Warning: Data path {SYNTHETIC_JOURNALS_PATH} does not exist")
        return dataset_files
    
    # Find all audio files and match with transcription files
    for audio_file in SYNTHETIC_JOURNALS_PATH.glob("journal_*.m4a"):
        # Find corresponding transcription file
        base_name = audio_file.stem  # e.g., 'journal_000'
        transcription_file = SYNTHETIC_JOURNALS_PATH / f"{base_name}.txt"
        
        if transcription_file.exists():
            dataset_files.append({
                'audio_file': str(audio_file),
                'transcription_file': str(transcription_file),
                'sample_id': base_name
            })
        else:
            print(f"Warning: No transcription found for {audio_file}")
    
    return dataset_files

def ensure_output_dirs():
    """Create output directories if they don't exist."""
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_OUTPUT_DIR.mkdir(exist_ok=True)
    EVALUATION_OUTPUT_DIR.mkdir(exist_ok=True)