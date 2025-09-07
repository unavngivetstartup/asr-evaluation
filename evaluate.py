"""ASR model evaluation script."""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Callable
import jiwer
from config import get_dataset_files, ensure_output_dirs, MODELS_OUTPUT_DIR
from schemas import TranscriptionResult, EvaluationMetrics, ModelEvaluationReport

def calculate_metrics(predictions: List[str], references: List[str]) -> dict:
    """Calculate evaluation metrics."""
    # Word Error Rate
    wer = jiwer.wer(references, predictions)
    
    # Character Error Rate  
    cer = jiwer.cer(references, predictions)
    
    # Exact match accuracy
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
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

def evaluate_model(
    model_name: str,
    model_version: str,
    transcription_function: Callable[[str], tuple],
    dataset_name: str = "synthetic_journals"
) -> ModelEvaluationReport:
    """
    Evaluate an ASR model on the dataset.
    
    Args:
        model_name: Name of the model (e.g., 'whisper')
        model_version: Version of the model (e.g., 'base', 'large-v2')
        transcription_function: Function that takes audio_file path and returns (transcription, confidence, processing_time)
        dataset_name: Name of the dataset being evaluated
    
    Returns:
        ModelEvaluationReport with complete evaluation results
    """
    ensure_output_dirs()
    
    dataset_files = get_dataset_files()
    if not dataset_files:
        raise ValueError("No dataset files found")
    
    print(f"Evaluating {model_name}-{model_version} on {len(dataset_files)} samples...")
    
    individual_results = []
    predictions = []
    references = []
    processing_times = []
    confidence_scores = []
    
    for file_info in dataset_files:
        print(f"Processing {file_info['sample_id']}...")
        
        # Load ground truth
        ground_truth = load_ground_truth(file_info['transcription_file'])
        references.append(ground_truth)
        
        # Run transcription
        start_time = time.time()
        try:
            transcription, confidence, model_processing_time = transcription_function(file_info['audio_file'])
            processing_time = model_processing_time or (time.time() - start_time)
        except Exception as e:
            print(f"Error processing {file_info['sample_id']}: {e}")
            transcription = ""
            confidence = None
            processing_time = time.time() - start_time
        
        predictions.append(transcription)
        processing_times.append(processing_time)
        if confidence is not None:
            confidence_scores.append(confidence)
        
        # Store individual result
        result = TranscriptionResult(
            sample_id=file_info['sample_id'],
            audio_file=file_info['audio_file'],
            model_name=model_name,
            model_version=model_version,
            transcription=transcription,
            confidence_score=confidence,
            processing_time_seconds=processing_time,
            timestamp=datetime.now()
        )
        individual_results.append(result)
    
    # Calculate metrics
    metrics_dict = calculate_metrics(predictions, references)
    
    metrics = EvaluationMetrics(
        model_name=model_name,
        model_version=model_version,
        word_error_rate=metrics_dict['word_error_rate'],
        character_error_rate=metrics_dict['character_error_rate'],
        exact_match_accuracy=metrics_dict['exact_match_accuracy'],
        avg_confidence_score=sum(confidence_scores) / len(confidence_scores) if confidence_scores else None,
        avg_processing_time=sum(processing_times) / len(processing_times),
        total_samples=len(dataset_files)
    )
    
    # Create evaluation report
    report = ModelEvaluationReport(
        model_name=model_name,
        model_version=model_version,
        dataset_name=dataset_name,
        evaluation_date=datetime.now(),
        metrics=metrics,
        individual_results=individual_results
    )
    
    # Save report
    report_filename = f"{model_name}_{model_version}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = MODELS_OUTPUT_DIR / report_filename
    report.save_to_json(str(report_path))
    
    print(f"Evaluation complete. Results saved to {report_path}")
    print(f"WER: {metrics.word_error_rate:.4f}")
    print(f"CER: {metrics.character_error_rate:.4f}")
    print(f"Exact Match: {metrics.exact_match_accuracy:.4f}")
    
    return report

# Example usage function for Whisper
def whisper_transcription_example(audio_file: str) -> tuple:
    """
    Example transcription function for Whisper.
    Replace this with actual Whisper implementation.
    """
    # This is a placeholder - replace with actual Whisper code
    # import whisper
    # model = whisper.load_model("base")
    # result = model.transcribe(audio_file)
    # return result["text"], None, None
    
    return "placeholder transcription", None, 1.0

if __name__ == "__main__":
    # Example evaluation run
    report = evaluate_model(
        model_name="whisper",
        model_version="base",
        transcription_function=whisper_transcription_example
    )