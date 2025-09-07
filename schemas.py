"""Data schemas for ASR evaluation results."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

@dataclass
class TranscriptionResult:
    """Single model transcription result."""
    sample_id: str
    audio_file: str
    model_name: str
    model_version: str
    transcription: str
    confidence_score: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'sample_id': self.sample_id,
            'audio_file': self.audio_file,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'transcription': self.transcription,
            'confidence_score': self.confidence_score,
            'processing_time_seconds': self.processing_time_seconds,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single model on a dataset."""
    model_name: str
    model_version: str
    word_error_rate: float
    character_error_rate: float
    exact_match_accuracy: float
    avg_confidence_score: Optional[float] = None
    avg_processing_time: Optional[float] = None
    total_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'word_error_rate': self.word_error_rate,
            'character_error_rate': self.character_error_rate,
            'exact_match_accuracy': self.exact_match_accuracy,
            'avg_confidence_score': self.avg_confidence_score,
            'avg_processing_time': self.avg_processing_time,
            'total_samples': self.total_samples
        }

@dataclass
class ModelEvaluationReport:
    """Complete evaluation report for a model."""
    model_name: str
    model_version: str
    dataset_name: str
    evaluation_date: datetime
    metrics: EvaluationMetrics
    individual_results: List[TranscriptionResult]
    
    def save_to_json(self, filepath: str):
        """Save report to JSON file."""
        report_data = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'dataset_name': self.dataset_name,
            'evaluation_date': self.evaluation_date.isoformat(),
            'metrics': self.metrics.to_dict(),
            'individual_results': [result.to_dict() for result in self.individual_results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'ModelEvaluationReport':
        """Load report from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = EvaluationMetrics(**data['metrics'])
        
        individual_results = []
        for result_data in data['individual_results']:
            if result_data['timestamp']:
                result_data['timestamp'] = datetime.fromisoformat(result_data['timestamp'])
            individual_results.append(TranscriptionResult(**result_data))
        
        return cls(
            model_name=data['model_name'],
            model_version=data['model_version'],
            dataset_name=data['dataset_name'],
            evaluation_date=datetime.fromisoformat(data['evaluation_date']),
            metrics=metrics,
            individual_results=individual_results
        )