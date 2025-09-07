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
        # Use audio_file parameter to avoid warning
        _ = audio_file
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


class VoxtralModel(ASRModel):
    """Mistral Voxtral multimodal ASR model implementation."""
    
    def __init__(self, model_size: str = "mini"):
        if model_size == "mini":
            version = "mini-3b-2507"
            self.repo_id = "mistralai/Voxtral-Mini-3B-2507"
        elif model_size == "small":
            version = "small-24b-2507"
            self.repo_id = "mistralai/Voxtral-Small-24B-2507"
        else:
            raise ValueError(f"Unknown Voxtral model size: {model_size}. Available: 'mini', 'small'")
            
        super().__init__("voxtral", version)
        self._model = None
        self._processor = None
        self.model_size = model_size
    
    def _load_model(self):
        """Lazy load the Voxtral model."""
        if self._model is None:
                import torch
                from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device
                
                device = infer_device()
                
                # Load processor and model using default cache
                self._processor = AutoProcessor.from_pretrained(self.repo_id)
                self._model = VoxtralForConditionalGeneration.from_pretrained(
                    self.repo_id, 
                    dtype=torch.bfloat16, 
                    device_map=device
                )
                
                self.device = device

    
    def _convert_to_wav(self, audio_file: str) -> str:
        """Convert audio file to WAV format if needed."""
        import tempfile
        
        # If already WAV, return as is
        if audio_file.lower().endswith('.wav'):
            return audio_file
            
        try:
            import librosa
            import soundfile as sf
            
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            # Load and convert audio
            audio_array, sr = librosa.load(audio_file, sr=None)
            sf.write(temp_wav_path, audio_array, sr)
            
            return temp_wav_path
            
        except ImportError:
            raise ImportError("librosa and soundfile are required for audio conversion. "
                            "Install with: pip install librosa soundfile")
    
    def transcribe(self, audio_file: str) -> Tuple[str, Optional[float], float]:
        """Transcribe audio using Voxtral."""
        self._load_model()
        
        start_time = time.time()
        
        try:
            import torch
            import os
            
            # Convert to WAV if needed
            wav_file = self._convert_to_wav(audio_file)
            temp_created = wav_file != audio_file
            
            try:
                # Create conversation format expected by Voxtral
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio",
                                "url": wav_file,
                            },
                            {"type": "text", "text": "You are an helpful AI assistant tasked with transcribing a speech file in Danish from a medical recording."},
                        ],
                    }
                ]
                
                # Process with the model
                inputs = self._processor.apply_chat_template(conversation)
                inputs = inputs.to(self.device, dtype=torch.bfloat16)
                
                # Generate transcription
                outputs = self._model.generate(**inputs, max_new_tokens=500)
                decoded_outputs = self._processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                
                transcription = decoded_outputs[0].strip()
                processing_time = time.time() - start_time
                
                return transcription, None, processing_time
                
            finally:
                # Clean up temporary WAV file if created
                if temp_created and os.path.exists(wav_file):
                    os.remove(wav_file)
                    
        except Exception as e:
            raise RuntimeError(f"Voxtral transcription failed: {e}")


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
    # Voxtral models
    "voxtral-mini": lambda: VoxtralModel("mini"),
    "voxtral-small": lambda: VoxtralModel("small"),
}

def get_model(model_id: str) -> ASRModel:
    """Get a model instance by ID."""
    if model_id not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model ID: {model_id}. Available models: {available}")
    
    return MODEL_REGISTRY[model_id]()