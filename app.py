"""Streamlit app for visualizing ASR evaluation results."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from config import MODELS_OUTPUT_DIR, ensure_output_dirs
from schemas import ModelEvaluationReport

def load_all_evaluation_reports() -> list:
    """Load all evaluation reports from the results directory."""
    ensure_output_dirs()
    
    reports = []
    for json_file in MODELS_OUTPUT_DIR.glob("*.json"):
        try:
            report = ModelEvaluationReport.load_from_json(str(json_file))
            reports.append(report)
        except Exception as e:
            st.error(f"Could not load {json_file.name}: {e}")
    
    return reports

def format_metric(value, format_type="percent"):
    """Format metrics for display."""
    if value is None:
        return "—"
    
    if format_type == "percent":
        return f"{value:.2%}"
    elif format_type == "decimal":
        return f"{value:.3f}"
    elif format_type == "time":
        return f"{value:.2f}s"
    
    return str(value)

def create_metrics_dataframe(reports: list) -> pd.DataFrame:
    """Create a clean DataFrame for model comparison."""
    data = []
    
    for report in reports:
        data.append({
            'Model': f"{report.model_name}-{report.model_version}",
            'WER': report.metrics.word_error_rate,
            'CER': report.metrics.character_error_rate,
            'Exact Match': report.metrics.exact_match_accuracy,
            'Avg Time (s)': report.metrics.avg_processing_time,
            'Samples': report.metrics.total_samples,
            'Date': report.evaluation_date.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(data)

def get_best_model(reports: list) -> ModelEvaluationReport:
    """Get the best performing model based on lowest WER."""
    return min(reports, key=lambda r: r.metrics.word_error_rate)

def main():
    st.set_page_config(
        page_title="ASR Evaluation",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load reports
    reports = load_all_evaluation_reports()
    
    if not reports:
        st.error("**No evaluation results found**")
        st.info("Run evaluations using `python evaluate.py` first")
        return
    
    # Header with clean metrics
    st.title("ASR Model Evaluation")
    st.caption(f"Comparing {len(reports)} model{'s' if len(reports) != 1 else ''} on speech recognition tasks")
    
    # Best model highlight
    if len(reports) > 1:
        best_model = get_best_model(reports)
        st.success(f"**Best performing model:** {best_model.model_name}-{best_model.model_version} "
                  f"(WER: {format_metric(best_model.metrics.word_error_rate, 'decimal')})")
    
    st.divider()
    
    # Main comparison table
    metrics_df = create_metrics_dataframe(reports)
    
    # Style the dataframe
    styled_df = metrics_df.style.format({
        'WER': lambda x: format_metric(x, 'decimal'),
        'CER': lambda x: format_metric(x, 'decimal'),
        'Exact Match': lambda x: format_metric(x, 'percent'),
        'Avg Time (s)': lambda x: format_metric(x, 'time') if x else "—"
    }).background_gradient(subset=['WER', 'CER'], cmap='RdYlGn_r')
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "WER": st.column_config.NumberColumn("Word Error Rate", help="Lower is better"),
            "CER": st.column_config.NumberColumn("Character Error Rate", help="Lower is better"),
            "Exact Match": st.column_config.NumberColumn("Exact Match Rate", help="Higher is better"),
            "Avg Time (s)": st.column_config.NumberColumn("Avg Processing Time"),
            "Samples": st.column_config.NumberColumn("Total Samples"),
            "Date": st.column_config.DateColumn("Evaluation Date")
        }
    )
    
    # Model details in sidebar
    with st.sidebar:
        st.header("Model Details")
        
        model_names = [f"{r.model_name}-{r.model_version}" for r in reports]
        selected = st.selectbox("Select model:", model_names, key="model_select")
        
        if selected:
            report = next(r for r in reports if f"{r.model_name}-{r.model_version}" == selected)
            
            st.metric("Word Error Rate", format_metric(report.metrics.word_error_rate, 'decimal'))
            st.metric("Character Error Rate", format_metric(report.metrics.character_error_rate, 'decimal'))
            st.metric("Exact Match", format_metric(report.metrics.exact_match_accuracy, 'percent'))
            
            if report.metrics.avg_processing_time:
                st.metric("Avg Processing Time", format_metric(report.metrics.avg_processing_time, 'time'))
            
            st.caption(f"Evaluated on {report.evaluation_date.strftime('%B %d, %Y at %H:%M')}")
            
            # Individual results
            if st.checkbox("Show individual results"):
                st.subheader("Sample Results")
                for result in report.individual_results:
                    with st.expander(f"Sample: {result.sample_id}"):
                        st.text_area("Transcription:", result.transcription, height=60, disabled=True)
                        if result.processing_time_seconds:
                            st.caption(f"Processing time: {result.processing_time_seconds:.2f}s")

if __name__ == "__main__":
    main()