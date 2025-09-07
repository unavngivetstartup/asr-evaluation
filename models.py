"""ASR model interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import time
import os

# Set deterministic behavior for reproducible results
os.environ['PYTHONHASHSEED'] = '0'

class ASRModel(ABC):
    """Abstract base class for ASR models."""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
    
    @abstractmethod
    def transcribe(self, audio_file: str) -> Tuple[str, Optional[float], float]:
        """
        Transcribe audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Tuple of (transcription, confidence_score, processing_time)
        """
        pass
    
    @property
    def model_id(self) -> str:
        """Get model identifier string."""
        return f"{self.name}-{self.version}"

class WhisperModel(ASRModel):
    """OpenAI Whisper model implementation."""
    
    def __init__(self, version: str = "base", language: str = "da"):
        super().__init__("whisper", version)
        self._model = None
        self.language = language
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.version)
    
    def transcribe(self, audio_file: str) -> Tuple[str, Optional[float], float]:
        """Transcribe audio using Whisper."""
        self._load_model()
        
        start_time = time.time()
        
        # Use deterministic settings for reproducible results
        result = self._model.transcribe(
            audio_file, 
            language=self.language,
            temperature=0  # Only change: greedy decoding for determinism
        )
        processing_time = time.time() - start_time
        
        # Whisper doesn't provide confidence scores in the basic API
        transcription = result["text"].strip()
        
        return transcription, None, processing_time

class MockModel(ASRModel):
    """Mock model for testing purposes."""
    
    def __init__(self, name: str = "mock", version: str = "v1", delay: float = 1.0):
        super().__init__(name, version)
        self.delay = delay
    
    def transcribe(self, audio_file: str) -> Tuple[str, Optional[float], float]:
        """Mock transcription with configurable delay."""
        time.sleep(self.delay)
        return "placeholder transcription", 0.95, self.delay

class HviskeV2Model(ASRModel):
    """syvai/hviske-v2 Danish ASR model implementation."""
    
    def __init__(self):
        super().__init__("hviske", "v2")
        self._pipeline = None
    
    def _load_model(self):
        """Lazy load the Hviske-v2 model."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                import torch
                
                # Set seeds for reproducible results
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Use pipeline for simplicity and robustness
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="syvai/hviske-v2",
                    torch_dtype=torch.float32,
                    device=device
                )
                
            except ImportError:
                raise ImportError(
                    "HuggingFace transformers library not found. "
                    "Install with: pip install transformers torch"
                )
    
    def transcribe(self, audio_file: str) -> Tuple[str, Optional[float], float]:
        """Transcribe audio using Hviske-v2."""
        self._load_model()
        
        start_time = time.time()
        
        try:
            import librosa
            import soundfile as sf
            import tempfile
            import os
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav_path = temp_file.name
            
            try:
                # Load and convert audio to WAV format
                audio_array, _ = librosa.load(audio_file, sr=16000)
                sf.write(temp_wav_path, audio_array, 16000)
                
                # Process with pipeline using temporary WAV file
                # Handle long-form audio (>30 seconds) by returning timestamps
                result = self._pipeline(temp_wav_path, return_timestamps=True)
                transcription = result["text"].strip()
                processing_time = time.time() - start_time
                
                return transcription, None, processing_time
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
            
        except ImportError as e:
            if "soundfile" in str(e):
                raise ImportError("soundfile is required. Install with: pip install soundfile")
            else:
                raise ImportError("librosa is required. Install with: pip install librosa")
        except Exception as e:
            raise RuntimeError(f"Hviske-v2 transcription failed: {e}")


# Model registry for easy access
MODEL_REGISTRY = {
    "whisper-tiny": lambda: WhisperModel("tiny"),
    "whisper-base": lambda: WhisperModel("base"),
    "whisper-small": lambda: WhisperModel("small"),
    "whisper-medium": lambda: WhisperModel("medium"),
    "whisper-large": lambda: WhisperModel("large"),
    "whisper-large-v2": lambda: WhisperModel("large-v2"),
    "whisper-large-v3": lambda: WhisperModel("large-v3"),
    "whisper-turbo": lambda: WhisperModel("turbo"),
    "mock": lambda: MockModel(),
    # Danish-specific models
    "hviske-v2": lambda: HviskeV2Model(),
}

def get_model(model_id: str) -> ASRModel:
    """Get a model instance by ID."""
    if model_id not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model ID: {model_id}. Available models: {available}")
    
    return MODEL_REGISTRY[model_id]()