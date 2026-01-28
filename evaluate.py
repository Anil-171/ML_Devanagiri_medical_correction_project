#!/usr/bin/env python3
"""
Evaluation script for Hindi Medical Correction Model

This script evaluates the trained model on the held-out eval.csv file
and reports various metrics including:
- Exact match percentage
- Character Error Rate (CER)
- Edit distance statistics
- Failure analysis

Usage:
    python evaluate.py --model_path ./model_output/final_model --eval_file eval.csv
"""

import os
import argparse
import json
import csv
import time
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
import editdistance

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Metrics Functions
# ==============================================================================

def compute_exact_match(predictions: List[str], targets: List[str]) -> float:
    """Compute exact match percentage."""
    matches = sum(p.strip() == t.strip() for p, t in zip(predictions, targets))
    return matches / len(predictions) * 100


def compute_edit_distance(pred: str, target: str) -> int:
    """Compute Levenshtein edit distance."""
    return editdistance.eval(pred, target)


def compute_character_error_rate(pred: str, target: str) -> float:
    """
    Compute Character Error Rate (CER).
    CER = edit_distance / len(target)
    """
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return compute_edit_distance(pred, target) / len(target)


def compute_word_error_rate(pred: str, target: str) -> float:
    """
    Compute Word Error Rate (WER).
    WER = word_edit_distance / num_words_in_target
    """
    pred_words = pred.split()
    target_words = target.split()
    
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    return editdistance.eval(pred_words, target_words) / len(target_words)


def analyze_error_types(pred: str, target: str) -> Dict:
    """Analyze types of errors between prediction and target."""
    errors = {
        'substitutions': 0,
        'insertions': 0,
        'deletions': 0
    }
    
    # Simple character-level analysis
    pred_chars = list(pred)
    target_chars = list(target)
    
    # Using dynamic programming for alignment would be more accurate,
    # but for simplicity we use length differences
    len_diff = len(pred_chars) - len(target_chars)
    
    if len_diff > 0:
        errors['insertions'] = len_diff
    elif len_diff < 0:
        errors['deletions'] = abs(len_diff)
    
    # Count differences in overlapping region
    min_len = min(len(pred_chars), len(target_chars))
    errors['substitutions'] = sum(p != t for p, t in zip(pred_chars[:min_len], target_chars[:min_len]))
    
    return errors


# ==============================================================================
# Evaluation Class
# ==============================================================================

class ModelEvaluator:
    """Evaluator for Hindi Medical Correction Model."""
    
    def __init__(self, model_path: str, device: str = None):
        """Initialize evaluator with model."""
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Model statistics
        self.model_params = sum(p.numel() for p in self.model.parameters())
        self.model_size_mb = self.model_params * 4 / (1024 * 1024)  # Assuming float32
        
        logger.info(f"Model loaded: {self.model_params:,} parameters ({self.model_size_mb:.1f} MB)")
    
    def predict(self, text: str, num_beams: int = 5) -> str:
        """Generate correction for a single input."""
        # Add prefix for T5-style models (must match training)
        prefixed_text = "correct: " + text
        
        inputs = self.tokenizer(
            prefixed_text,
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=num_beams,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, num_beams: int = 5) -> List[str]:
        """Generate corrections for a batch of inputs."""
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Add prefix for T5-style models (must match training)
            prefixed_batch = ["correct: " + t for t in batch]
            
            inputs = self.tokenizer(
                prefixed_batch,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=128,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
        
        return predictions
    
    def evaluate(self, eval_df: pd.DataFrame, batch_size: int = 32) -> Dict:
        """
        Evaluate model on a dataframe.
        
        Args:
            eval_df: DataFrame with 'Hindi_raw' and 'expected_outputs' columns
        
        Returns:
            Dictionary with evaluation metrics
        """
        inputs = eval_df['Hindi_raw'].tolist()
        targets = eval_df['expected_outputs'].tolist()
        
        logger.info(f"Evaluating on {len(inputs)} samples...")
        
        # Measure inference time
        start_time = time.time()
        predictions = []
        
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating predictions"):
            batch = inputs[i:i + batch_size]
            batch_preds = self.predict_batch(batch, batch_size=len(batch))
            predictions.extend(batch_preds)
        
        total_time = time.time() - start_time
        avg_latency = (total_time / len(inputs)) * 1000  # ms per sample
        
        # Compute metrics
        exact_match = compute_exact_match(predictions, targets)
        
        # Compute per-sample metrics
        edit_distances = []
        cers = []
        wers = []
        error_analysis = []
        
        for pred, target in zip(predictions, targets):
            ed = compute_edit_distance(pred, target)
            cer = compute_character_error_rate(pred, target)
            wer = compute_word_error_rate(pred, target)
            
            edit_distances.append(ed)
            cers.append(cer)
            wers.append(wer)
            error_analysis.append(analyze_error_types(pred, target))
        
        # Aggregate metrics
        metrics = {
            "exact_match_percentage": round(exact_match, 2),
            "avg_edit_distance": round(np.mean(edit_distances), 2),
            "std_edit_distance": round(np.std(edit_distances), 2),
            "avg_character_error_rate": round(np.mean(cers) * 100, 2),  # as percentage
            "avg_word_error_rate": round(np.mean(wers) * 100, 2),  # as percentage
            "total_samples": len(inputs),
            "correct_samples": int(exact_match * len(inputs) / 100),
            "inference_time_seconds": round(total_time, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "model_params": self.model_params,
            "model_size_mb": round(self.model_size_mb, 2),
            "device": self.device
        }
        
        # Find failures
        failures = []
        over_corrections = []
        
        for i, (inp, pred, target) in enumerate(zip(inputs, predictions, targets)):
            if pred.strip() != target.strip():
                failures.append({
                    "index": i,
                    "input": inp,
                    "predicted": pred,
                    "expected": target,
                    "edit_distance": edit_distances[i],
                    "cer": round(cers[i] * 100, 2)
                })
            
            # Check for over-correction (input was already correct but model changed it)
            if inp.strip() == target.strip() and pred.strip() != target.strip():
                over_corrections.append({
                    "index": i,
                    "input": inp,
                    "predicted": pred,
                    "expected": target
                })
        
        metrics["num_failures"] = len(failures)
        metrics["num_over_corrections"] = len(over_corrections)
        
        return metrics, predictions, failures, over_corrections


def load_eval_csv(csv_path: str) -> pd.DataFrame:
    """Load and preprocess eval.csv."""
    logger.info(f"Loading evaluation data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check column names and normalize
    if 'Hindi_raw' in df.columns and 'expected_outputs' in df.columns:
        pass  # Already correct
    elif 'raw_input' in df.columns and 'expected_output' in df.columns:
        df = df.rename(columns={'raw_input': 'Hindi_raw', 'expected_output': 'expected_outputs'})
    else:
        # Try first two columns
        cols = df.columns.tolist()
        df = df.rename(columns={cols[0]: 'Hindi_raw', cols[1]: 'expected_outputs'})
    
    # Remove rows with empty values
    df = df.dropna(subset=['Hindi_raw', 'expected_outputs'])
    df = df[df['Hindi_raw'].str.strip() != '']
    df = df[df['expected_outputs'].str.strip() != '']
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} valid samples")
    
    return df


def generate_report(metrics: Dict, failures: List[Dict], over_corrections: List[Dict]) -> str:
    """Generate a human-readable evaluation report."""
    
    report = []
    report.append("=" * 70)
    report.append("HINDI MEDICAL CORRECTION MODEL - EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Overall metrics
    report.append("OVERALL METRICS:")
    report.append("-" * 40)
    report.append(f"  Total Samples:           {metrics['total_samples']}")
    report.append(f"  Exact Match:             {metrics['exact_match_percentage']:.2f}%")
    report.append(f"  Correct Samples:         {metrics['correct_samples']}")
    report.append(f"  Failed Samples:          {metrics['num_failures']}")
    report.append("")
    
    # Error metrics
    report.append("ERROR METRICS:")
    report.append("-" * 40)
    report.append(f"  Avg Edit Distance:       {metrics['avg_edit_distance']:.2f} (±{metrics['std_edit_distance']:.2f})")
    report.append(f"  Character Error Rate:    {metrics['avg_character_error_rate']:.2f}%")
    report.append(f"  Word Error Rate:         {metrics['avg_word_error_rate']:.2f}%")
    report.append("")
    
    # Inference metrics
    report.append("INFERENCE METRICS:")
    report.append("-" * 40)
    report.append(f"  Total Inference Time:    {metrics['inference_time_seconds']:.2f} seconds")
    report.append(f"  Average Latency:         {metrics['avg_latency_ms']:.2f} ms/sample")
    report.append(f"  Device:                  {metrics['device']}")
    report.append("")
    
    # Model info
    report.append("MODEL INFO:")
    report.append("-" * 40)
    report.append(f"  Parameters:              {metrics['model_params']:,}")
    report.append(f"  Size:                    ~{metrics['model_size_mb']:.1f} MB")
    report.append("")
    
    # Failure analysis
    report.append("FAILURE ANALYSIS:")
    report.append("-" * 40)
    report.append(f"  Total Failures:          {len(failures)}")
    report.append(f"  Over-corrections:        {metrics['num_over_corrections']}")
    report.append("")
    
    # Sample failures
    if failures:
        report.append("EXAMPLE FAILURES (first 10):")
        report.append("-" * 40)
        for i, f in enumerate(failures[:10]):
            report.append(f"\n  Failure {i+1}:")
            report.append(f"    Input:    {f['input']}")
            report.append(f"    Expected: {f['expected']}")
            report.append(f"    Got:      {f['predicted']}")
            report.append(f"    ED: {f['edit_distance']}, CER: {f['cer']}%")
    
    # Over-corrections
    if over_corrections:
        report.append("\n" + "OVER-CORRECTIONS (already correct input was changed):")
        report.append("-" * 40)
        for i, oc in enumerate(over_corrections[:5]):
            report.append(f"\n  Over-correction {i+1}:")
            report.append(f"    Input (correct): {oc['input']}")
            report.append(f"    Model output:    {oc['predicted']}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hindi Medical Correction Model')
    parser.add_argument('--model_path', type=str, default='./model_output/final_model',
                       help='Path to the trained model')
    parser.add_argument('--eval_file', type=str, default='eval.csv',
                       help='Path to evaluation CSV file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load evaluation data
    eval_df = load_eval_csv(args.eval_file)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.device)
    
    # Run evaluation
    metrics, predictions, failures, over_corrections = evaluator.evaluate(
        eval_df, 
        batch_size=args.batch_size
    )
    
    # Generate and print report
    report = generate_report(metrics, failures, over_corrections)
    print(report)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save detailed results CSV
    results_df = pd.DataFrame({
        'Hindi_raw': eval_df['Hindi_raw'],
        'expected_outputs': eval_df['expected_outputs'],
        'model_prediction': predictions,
        'is_correct': [p.strip() == t.strip() for p, t in zip(predictions, eval_df['expected_outputs'])]
    })
    
    results_path = os.path.join(args.output_dir, 'Evaluate_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    logger.info(f"Detailed results saved to {results_path}")
    
    # Save report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    # Save failures for analysis
    failures_path = os.path.join(args.output_dir, 'failures.json')
    with open(failures_path, 'w', encoding='utf-8') as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    logger.info(f"Failures saved to {failures_path}")
    
    print(f"\n{'='*50}")
    print("SUMMARY:")
    print(f"  Exact Match: {metrics['exact_match_percentage']:.2f}%")
    print(f"  Avg CER: {metrics['avg_character_error_rate']:.2f}%")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
