# ASR Evaluation Project

This project evaluates Automatic Speech Recognition (ASR) tools and models, including scripts for running evaluations and a Streamlit application for visualizing results.

## Project Structure

- **Evaluation Scripts**: Run evaluations using different speech-to-text models (e.g., Whisper)
- **Streamlit App**: Display model performance in a nice table format

## Data Structure

**External Data Location**: `/homes/hinge/Projects/data-samples/data/synthetic_journals/`
- Audio files: `journal_XXX.m4a` 
- Ground truth transcriptions: `journal_XXX.txt`

**Output Structure**:
- `results/models/`: JSON reports with transcription results and metrics
- Model outputs follow standardized schema with WER, CER, exact match accuracy
- Individual sample results include confidence scores and processing times

## Common Commands

```bash
# Install dependencies (using uv)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run evaluation (modify evaluate.py with your model implementation)
python evaluate.py

# Start Streamlit application
streamlit run app.py

# Create output directories
mkdir -p results/{models,evaluations}
```

## Model Performance

The project supports evaluation of various ASR models:
- OpenAI Whisper (various sizes)
- Other speech-to-text models

Results are displayed in an interactive table showing performance metrics across different models.