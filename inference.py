#!/usr/bin/env python3
"""
Inference script for Hindi Medical Correction Model

This script provides an interface for correcting noisy Hindi medical text
using the trained model.

Usage:
    python inference.py --model_path ./model_output/final_model --input "मुझे पेन में दर्द है"
    
Or as a module:
    from inference import HindiMedicalCorrector
    corrector = HindiMedicalCorrector("./model_output/final_model")
    result = corrector.correct("मुझे पेन में दर्द है")
"""

import os
import argparse
import time
import json
from typing import List, Dict, Optional, Tuple
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HindiMedicalCorrector:
    """
    Hindi Medical Text Correction Engine
    
    This class provides methods for correcting noisy Hindi medical sentences
    using a fine-tuned mT5 model.
    """
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 128
    ):
        """
        Initialize the corrector.
        
        Args:
            model_path: Path to the saved model directory
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            max_length: Maximum sequence length for generation
        """
        self.model_path = model_path
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Model info
        self.model_size = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {self.model_size:,}")
    
    def correct(
        self, 
        text: str,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> Dict:
        """
        Correct a single noisy Hindi sentence.
        
        Args:
            text: Input noisy text
            num_beams: Number of beams for beam search
            num_return_sequences: Number of alternative corrections to return
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        
        Returns:
            Dictionary containing:
                - corrected_hindi: Best correction
                - top_k_alternatives: List of alternatives with scores
                - confidence: Confidence score
                - latency_ms: Inference latency in milliseconds
        """
        start_time = time.time()
        
        # Add prefix for T5-style models (must match training)
        prefixed_text = "correct: " + text
        
        # Tokenize input
        inputs = self.tokenizer(
            prefixed_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate corrections
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.max_length,
                num_beams=num_beams,
                num_return_sequences=min(num_return_sequences, num_beams),
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True,
                do_sample=False if num_beams > 1 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode outputs
        sequences = outputs.sequences
        
        # Calculate sequence scores (log probabilities)
        if hasattr(outputs, 'sequences_scores'):
            scores = outputs.sequences_scores.cpu().numpy().tolist()
        else:
            # Fallback: use simple scoring
            scores = [0.0] * len(sequences)
        
        # Decode all sequences
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        # Calculate confidence (normalized score)
        if scores and scores[0] != 0:
            # Convert log probability to probability-like score
            confidence = min(1.0, max(0.0, 1.0 + scores[0] / 10))
        else:
            confidence = 0.8  # Default confidence
        
        # Prepare alternatives
        alternatives = []
        for i, (seq, score) in enumerate(zip(decoded, scores)):
            alternatives.append({
                "text": seq,
                "score": float(score) if score else 0.0,
                "rank": i + 1
            })
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            "input": text,
            "corrected_hindi": decoded[0] if decoded else text,
            "top_k_alternatives": alternatives,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2)
        }
        
        return result
    
    def correct_batch(
        self, 
        texts: List[str],
        num_beams: int = 5,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Correct a batch of noisy Hindi sentences.
        
        Args:
            texts: List of input noisy texts
            num_beams: Number of beams for beam search
            batch_size: Batch size for processing
        
        Returns:
            List of correction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Add prefix for T5-style models (must match training)
            prefixed_texts = ["correct: " + t for t in batch_texts]
            
            start_time = time.time()
            
            # Tokenize batch
            inputs = self.tokenizer(
                prefixed_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            batch_latency = (time.time() - start_time) * 1000
            per_sample_latency = batch_latency / len(batch_texts)
            
            for text, correction in zip(batch_texts, decoded):
                results.append({
                    "input": text,
                    "corrected_hindi": correction,
                    "confidence": 0.85,  # Simplified for batch
                    "latency_ms": round(per_sample_latency, 2)
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        return {
            "model_path": self.model_path,
            "model_size_params": self.model_size,
            "model_size_mb": round(self.model_size * 4 / (1024 * 1024), 2),  # Assuming float32
            "device": self.device,
            "max_length": self.max_length
        }
    
    def measure_inference_latency(self, text: str, num_runs: int = 10) -> Dict:
        """
        Measure inference latency over multiple runs.
        
        Args:
            text: Sample text for benchmarking
            num_runs: Number of inference runs
        
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        # Warmup
        _ = self.correct(text)
        
        # Measure
        for _ in range(num_runs):
            result = self.correct(text)
            latencies.append(result['latency_ms'])
        
        return {
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "num_runs": num_runs
        }


def interactive_mode(corrector: HindiMedicalCorrector):
    """Run interactive correction mode."""
    print("\n" + "=" * 60)
    print("Hindi Medical Text Correction - Interactive Mode")
    print("=" * 60)
    print("Enter Hindi text to correct. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            text = input("Input: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            result = corrector.correct(text, num_return_sequences=3)
            
            print(f"\nCorrected: {result['corrected_hindi']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Latency: {result['latency_ms']:.1f} ms")
            
            if len(result['top_k_alternatives']) > 1:
                print("\nAlternatives:")
                for alt in result['top_k_alternatives'][1:]:
                    print(f"  {alt['rank']}. {alt['text']} (score: {alt['score']:.4f})")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Hindi Medical Text Correction Inference')
    parser.add_argument('--model_path', type=str, default='./model_output/final_model',
                       help='Path to the trained model')
    parser.add_argument('--input', type=str, default=None,
                       help='Input text to correct')
    parser.add_argument('--input_file', type=str, default=None,
                       help='File with inputs (one per line)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results (JSON)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--num_beams', type=int, default=5,
                       help='Number of beams for beam search')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize corrector
    corrector = HindiMedicalCorrector(
        model_path=args.model_path,
        device=args.device
    )
    
    # Print model info
    info = corrector.get_model_info()
    print(f"\nModel Info:")
    print(f"  Parameters: {info['model_size_params']:,}")
    print(f"  Size: ~{info['model_size_mb']:.1f} MB")
    print(f"  Device: {info['device']}")
    
    if args.interactive:
        interactive_mode(corrector)
    elif args.input:
        # Single input
        result = corrector.correct(args.input, num_beams=args.num_beams, num_return_sequences=3)
        
        print(f"\nInput:     {result['input']}")
        print(f"Corrected: {result['corrected_hindi']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Latency: {result['latency_ms']:.1f} ms")
        
        if len(result['top_k_alternatives']) > 1:
            print("\nAlternatives:")
            for alt in result['top_k_alternatives']:
                print(f"  {alt['rank']}. {alt['text']} (score: {alt['score']:.4f})")
        
    elif args.input_file:
        # Batch processing
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"\nProcessing {len(texts)} inputs...")
        results = corrector.correct_batch(texts, num_beams=args.num_beams)
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output_file}")
        else:
            for r in results:
                print(f"Input: {r['input']}")
                print(f"Output: {r['corrected_hindi']}")
                print()
    else:
        # Demo with sample inputs
        print("\nDemo - Correcting sample inputs:")
        print("-" * 50)
        
        samples = [
            "मुझे पेन में दर्द है",
            "बुखाद आने पर बहुत ठंद लगती है",
            "खांबी के साथ हल्दा बुखार है",
            "मेरे गर्दद में दर्द है",
            "सीने में भारपन और जकदन लगती है"
        ]
        
        for text in samples:
            result = corrector.correct(text)
            print(f"Input:     {text}")
            print(f"Corrected: {result['corrected_hindi']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print()
        
        # Measure latency
        latency_stats = corrector.measure_inference_latency(samples[0])
        print(f"\nLatency Statistics (over {latency_stats['num_runs']} runs):")
        print(f"  Mean: {latency_stats['mean_latency_ms']:.1f} ms")
        print(f"  Min:  {latency_stats['min_latency_ms']:.1f} ms")
        print(f"  Max:  {latency_stats['max_latency_ms']:.1f} ms")


if __name__ == '__main__':
    main()
