#!/usr/bin/env python3
"""
Training script for Hindi Medical Correction Model

This script fine-tunes a pre-trained multilingual seq2seq model (mT5-small)
for correcting noisy Hindi medical text.

Model choice rationale:
- mT5-small: Multilingual T5 model that supports Hindi
- Sequence-to-sequence architecture is ideal for text correction
- Small model size (~300M params) for faster training and inference
- Good balance between accuracy and resource efficiency
"""

import os
import argparse
import logging
import time
from datetime import datetime
import json

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset, DatasetDict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    # Model settings
    MODEL_NAME = "google/mt5-small"  # Multilingual T5 small
    MAX_INPUT_LENGTH = 128
    MAX_TARGET_LENGTH = 128
    
    # Training settings - BEST PERFORMING (v1 settings)
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 5e-5  # Standard for fine-tuning
    NUM_EPOCHS = 5  # Sufficient convergence
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.0  # No smoothing for v1
    
    # Data settings
    TRAIN_SPLIT = 0.9
    VAL_SPLIT = 0.1
    
    # Paths
    OUTPUT_DIR = "./model_output"
    CHECKPOINT_DIR = "./checkpoints"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Custom Dataset Class
# ==============================================================================

class HindiCorrectionDataset(Dataset):
    """Custom PyTorch Dataset for Hindi text correction."""
    
    def __init__(self, data: pd.DataFrame, tokenizer, max_input_len: int, max_target_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize input (noisy text)
        input_encoding = self.tokenizer(
            row['noisy_input'],
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (clean text)
        target_encoding = self.tokenizer(
            row['clean_output'],
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


# ==============================================================================
# Data Preprocessing Functions
# ==============================================================================

def load_and_preprocess_data(csv_path: str, tokenizer, config: Config):
    """Load CSV and prepare datasets."""
    
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Basic cleaning
    df = df.dropna()
    df = df[df['noisy_input'].str.len() > 0]
    df = df[df['clean_output'].str.len() > 0]
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    train_size = int(len(df) * config.TRAIN_SPLIT)
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
    
    return train_df, val_df


def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    """Preprocess function for HuggingFace datasets."""
    
    inputs = examples['noisy_input']
    targets = examples['clean_output']
    
    # Add prefix for T5-style models to indicate task
    inputs = ["correct: " + inp for inp in inputs]
    
    # Tokenize inputs - no padding, let collator handle it
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=False
    )
    
    # Tokenize targets - no padding, let collator handle it
    # Use the tokenizer directly for targets
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding=False
    )
    
    # Set labels - collator will handle padding with -100
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs


# ==============================================================================
# Training Functions
# ==============================================================================

def train_with_trainer(train_df, val_df, config: Config):
    """Train using HuggingFace Trainer (recommended approach)."""
    
    logger.info(f"Loading model: {config.MODEL_NAME}")
    logger.info(f"Using device: {config.DEVICE}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    
    # Move model to device
    model = model.to(config.DEVICE)
    
    # Convert to HuggingFace datasets
    train_dataset = HFDataset.from_pandas(train_df[['noisy_input', 'clean_output']])
    val_dataset = HFDataset.from_pandas(val_df[['noisy_input', 'clean_output']])
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    
    def tokenize_fn(examples):
        return preprocess_function(
            examples, tokenizer, 
            config.MAX_INPUT_LENGTH, 
            config.MAX_TARGET_LENGTH
        )
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=['noisy_input', 'clean_output']
    )
    
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=['noisy_input', 'clean_output']
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        label_smoothing_factor=config.LABEL_SMOOTHING,  # V3: Better generalization
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=False,  # Disable mixed precision for stability
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard
        dataloader_num_workers=0,  # Use single process to avoid permission issues
        max_grad_norm=1.0,  # Gradient clipping for stability
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Debug: Print a sample from the dataset to verify data is correct
    logger.info("Debugging: Checking sample from training dataset...")
    sample = train_dataset[0]
    logger.info(f"Sample input_ids length: {len(sample['input_ids'])}")
    logger.info(f"Sample labels length: {len(sample['labels'])}")
    logger.info(f"Sample labels first 20: {sample['labels'][:20]}")
    logger.info(f"Pad token id: {tokenizer.pad_token_id}")
    logger.info(f"Learning rate from config: {config.LEARNING_RATE}")
    
    # Verify labels aren't all -100
    non_minus100 = sum(1 for l in sample['labels'] if l != -100)
    logger.info(f"Non -100 labels count: {non_minus100}")
    
    # Train
    logger.info("Starting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model
    final_model_path = os.path.join(config.OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"Model saved to {final_model_path}")
    
    # Save training metrics
    metrics = {
        "training_time_seconds": training_time,
        "train_loss": train_result.training_loss,
        "model_name": config.MODEL_NAME,
        "num_epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }
    
    with open(os.path.join(config.OUTPUT_DIR, "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return trainer, tokenizer


def train_custom_loop(train_df, val_df, config: Config):
    """Custom training loop for more control (alternative approach)."""
    
    logger.info(f"Loading model: {config.MODEL_NAME}")
    logger.info(f"Using device: {config.DEVICE}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    model = model.to(config.DEVICE)
    
    # Create datasets
    train_dataset = HindiCorrectionDataset(
        train_df, tokenizer, 
        config.MAX_INPUT_LENGTH, 
        config.MAX_TARGET_LENGTH
    )
    val_dataset = HindiCorrectionDataset(
        val_df, tokenizer,
        config.MAX_INPUT_LENGTH,
        config.MAX_TARGET_LENGTH
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if config.DEVICE == 'cuda' else None
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(config.DEVICE)
                attention_mask = batch['attention_mask'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config.OUTPUT_DIR, "best_model")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Saved best model with val_loss = {best_val_loss:.4f}")
    
    # Save final model
    final_path = os.path.join(config.OUTPUT_DIR, "final_model")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    return model, tokenizer


# ==============================================================================
# Evaluation during training
# ==============================================================================

def compute_metrics(eval_pred, tokenizer):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute exact match
    exact_matches = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
    exact_match_ratio = exact_matches / len(decoded_preds)
    
    return {
        "exact_match": exact_match_ratio
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Hindi Medical Correction Model')
    parser.add_argument('--data', type=str, default='dataset.csv',
                       help='Path to training data CSV')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./model_output',
                       help='Output directory for model')
    parser.add_argument('--use_trainer', action='store_true', default=True,
                       help='Use HuggingFace Trainer (recommended)')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.OUTPUT_DIR = args.output_dir
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Log configuration
    logger.info("=" * 50)
    logger.info("Training Configuration:")
    logger.info(f"  Model: {config.MODEL_NAME}")
    logger.info(f"  Device: {config.DEVICE}")
    logger.info(f"  Epochs: {config.NUM_EPOCHS}")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  Output Dir: {config.OUTPUT_DIR}")
    logger.info("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer for data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Load and preprocess data
    train_df, val_df = load_and_preprocess_data(args.data, tokenizer, config)
    
    # Train
    if args.use_trainer:
        trainer, tokenizer = train_with_trainer(train_df, val_df, config)
    else:
        model, tokenizer = train_custom_loop(train_df, val_df, config)
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {config.OUTPUT_DIR}/final_model")


if __name__ == '__main__':
    main()
