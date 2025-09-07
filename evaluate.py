"""ASR model evaluation script."""

import time
from datetime import datetime
from pathlib import Path
from typing import List
import jiwer
import string
import re
from config import get_dataset_files, ensure_output_dirs, MODELS_OUTPUT_DIR
from schemas import TranscriptionResult, EvaluationMetrics, ModelEvaluationReport
from models import ASRModel, get_model
from models import MODEL_REGISTRY
import glob

def normalize_text(text: str) -> str:
    """
    Normalize text for fair ASR evaluation.
    
    - Convert to lowercase
    - Remove punctuation
    - Normalize whitespace
    - Remove leading/trailing whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def calculate_metrics(predictions: List[str], references: List[str]) -> dict:
    """Calculate evaluation metrics with proper text normalization."""
    # Normalize all texts for fair comparison
    norm_predictions = [normalize_text(p) for p in predictions]
    norm_references = [normalize_text(r) for r in references]
    
    # Word Error Rate (normalized)
    wer = jiwer.wer(norm_references, norm_predictions)
    
    # Character Error Rate (normalized)
    cer = jiwer.cer(norm_references, norm_predictions)
    
    # Exact match accuracy (normalized)
    exact_matches = sum(1 for p, r in zip(norm_predictions, norm_references) if p == r)
    exact_match_accuracy = exact_matches / len(predictions) if predictions else 0
    
    return {
        'word_error_rate': wer,
        'character_error_rate': cer,
        'exact_match_accuracy': exact_match_accuracy
    }

def load_ground_truth(transcription_file: str) -> str:
    """Load ground truth transcription from file."""
    with open(transcription_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

def is_already_evaluated(model: ASRModel, dataset_name: str = "synthetic_journals") -> bool:
    """
    Check if a model has already been evaluated on the dataset.
    
    Args:
        model: ASRModel instance to check
        dataset_name: Name of the dataset
    
    Returns:
        True if evaluation already exists, False otherwise
    """
    # Look for existing evaluation files matching this model and dataset
    pattern = f"{model.name}_{model.version}_{dataset_name}_*.json"
    search_path = MODELS_OUTPUT_DIR / pattern
    
    existing_files = glob.glob(str(search_path))
    return len(existing_files) > 0

def get_latest_evaluation_path(model: ASRModel, dataset_name: str = "synthetic_journals") -> str:
    """
    Get the path to the most recent evaluation file for a model.
    
    Args:
        model: ASRModel instance
        dataset_name: Name of the dataset
    
    Returns:
        Path to the latest evaluation file, or None if not found
    """
    pattern = f"{model.name}_{model.version}_{dataset_name}_*.json"
    search_path = MODELS_OUTPUT_DIR / pattern
    
    existing_files = glob.glob(str(search_path))
    if not existing_files:
        return None
    
    # Return the most recent file (assuming timestamp in filename)
    return max(existing_files)

def evaluate_model(
    model: ASRModel,
    dataset_name: str = "synthetic_journals",
    force_reevaluate: bool = False
) -> ModelEvaluationReport:
    """
    Evaluate an ASR model on the dataset.
    
    Args:
        model: ASRModel instance to evaluate
        dataset_name: Name of the dataset being evaluated
        force_reevaluate: If True, skip existing evaluation check
    
    Returns:
        ModelEvaluationReport with complete evaluation results
    """
    # Check if already evaluated (unless forced)
    if not force_reevaluate and is_already_evaluated(model, dataset_name):
        latest_path = get_latest_evaluation_path(model, dataset_name)
        print(f"✅ {model.model_id} already evaluated. Loading existing results: {latest_path}")
        return ModelEvaluationReport.load_from_json(latest_path)
    ensure_output_dirs()
    
    dataset_files = get_dataset_files()
    if not dataset_files:
        raise ValueError("No dataset files found")
    
    print(f"Evaluating {model.model_id} on {len(dataset_files)} samples...")
    
    individual_results = []
    predictions = []
    references = []
    processing_times = []
    confidence_scores = []
    failed_samples = []
    
    for file_info in dataset_files:
        print(f"Processing {file_info['sample_id']}...")
        
        # Load ground truth
        ground_truth = load_ground_truth(file_info['transcription_file'])
        references.append(ground_truth)
        
        # Run transcription
        try:
            transcription, confidence, processing_time = model.transcribe(file_info['audio_file'])
        except Exception as e:
            print(f"Error processing {file_info['sample_id']}: {e}")
            failed_samples.append(file_info['sample_id'])
            transcription = ""
            confidence = None
            processing_time = 0.0
        
        predictions.append(transcription)
        processing_times.append(processing_time)
        if confidence is not None:
            confidence_scores.append(confidence)
        
        # Store individual result
        result = TranscriptionResult(
            sample_id=file_info['sample_id'],
            audio_file=file_info['audio_file'],
            model_name=model.name,
            model_version=model.version,
            transcription=transcription,
            confidence_score=confidence,
            processing_time_seconds=processing_time,
            timestamp=datetime.now()
        )
        individual_results.append(result)
    
    # Check if evaluation failed significantly
    failure_rate = len(failed_samples) / len(dataset_files)
    if failure_rate > 0.5:  # More than 50% failed
        print(f"❌ Evaluation failed: {len(failed_samples)}/{len(dataset_files)} samples failed ({failure_rate:.1%})")
        print(f"Failed samples: {', '.join(failed_samples)}")
        print("Results not saved due to high failure rate.")
        return None
    
    if failed_samples:
        print(f"⚠️  Warning: {len(failed_samples)} samples failed: {', '.join(failed_samples)}")
    
    # Calculate metrics
    metrics_dict = calculate_metrics(predictions, references)
    
    metrics = EvaluationMetrics(
        model_name=model.name,
        model_version=model.version,
        word_error_rate=metrics_dict['word_error_rate'],
        character_error_rate=metrics_dict['character_error_rate'],
        exact_match_accuracy=metrics_dict['exact_match_accuracy'],
        avg_confidence_score=sum(confidence_scores) / len(confidence_scores) if confidence_scores else None,
        avg_processing_time=sum(processing_times) / len(processing_times),
        total_samples=len(dataset_files)
    )
    
    # Create evaluation report
    report = ModelEvaluationReport(
        model_name=model.name,
        model_version=model.version,
        dataset_name=dataset_name,
        evaluation_date=datetime.now(),
        metrics=metrics,
        individual_results=individual_results
    )
    
    # Save report
    report_filename = f"{model.name}_{model.version}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = MODELS_OUTPUT_DIR / report_filename
    report.save_to_json(str(report_path))
    
    print(f"Evaluation complete. Results saved to {report_path}")
    print(f"WER: {metrics.word_error_rate:.4f}")
    print(f"CER: {metrics.character_error_rate:.4f}")
    print(f"Exact Match: {metrics.exact_match_accuracy:.4f}")
    
    return report

if __name__ == "__main__":
    # Example evaluation run - using the modular model system
    for model_id in MODEL_REGISTRY.keys():
        model = get_model(model_id)
        
        # Check if already evaluated
        if is_already_evaluated(model):
            latest_path = get_latest_evaluation_path(model)
            print(f"✅ {model_id} already evaluated. Results: {latest_path}")
            continue
        
        print(f"🚀 Starting evaluation for {model_id}...")
        report = evaluate_model(model)
        if report is None:
            print(f"❌ Skipping {model_id} due to evaluation failure\n")
            continue
        
        print(f"✅ {model_id} evaluation completed\n")