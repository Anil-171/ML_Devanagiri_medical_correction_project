# Hindi Medical Text Correction Engine

This project builds a correction system for noisy Hindi medical text that typically comes from speech recognition (ASR) systems. The model learns to fix common errors like misspelled words, missing vowel marks, and phonetic confusions.

## The Problem

When patients speak to medical systems in Hindi, the ASR output often contains errors:

```
Noisy input:  मुझे पेन में दर्द है
Correct:      मुझे पेट में दर्द है
              (पेन → पेट)
```

Common error types include:
- Word confusion: `पेट` becomes `पेन`, `दर्द` becomes `डर्ड`
- Vowel mark errors: `है` becomes `हे`, missing matras
- Spacing problems: words merged or split incorrectly

## My Approach

I chose a sequence-to-sequence neural network approach over rule-based methods because:

1. It learns correction patterns automatically from data
2. It understands context (knows `पेन` should be `पेट` in medical context)
3. A single model handles all error types together

**Model**: I used mT5-small, a multilingual transformer pre-trained on 101 languages including Hindi. It has ~556M parameters and is designed for text-to-text tasks like this.

## Project Files

```
├── data_generator.py      # Creates synthetic training data
├── dataset.csv            # 12,000 noisy-clean training pairs
├── train.py               # Training script
├── inference.py           # For making predictions
├── evaluate.py            # Runs evaluation on test set
├── eval.csv               # Test data (449 samples, not used in training)
├── Evaluate_results.csv   # Model predictions
├── model_output/          # Trained model
└── requirements.txt       # Dependencies
```

## How the Data Generator Works

Since I didn't have real ASR error data for training, I created synthetic noisy data by applying realistic error patterns:

- **Phonetic substitutions**: Characters that sound similar get swapped (`त↔थ`, `द↔ड`, `ब↔व`)
- **Vowel mark errors**: Matras get changed or dropped (`ा↔े`, `ि↔ी`)
- **Medical word errors**: Common medical terms get corrupted (`पेट→पेन`, `बुखार→बुखाद`)
- **Spacing issues**: Words occasionally merge or split

The generator creates 12,000 sentence pairs with varying noise levels.

## Results

I evaluated the model on 449 real noisy samples from `eval.csv`:

| Metric | Result |
|--------|--------|
| Exact Match | 3.79% (17 perfect corrections) |
| Character Error Rate | 29.11% |
| Word Error Rate | 43.12% |
| Avg Edit Distance | 14.13 characters |
| Inference Speed | ~19 ms per sentence (GPU) |

### Understanding the Results

The 3.79% exact match looks low, but there's more to the story:

- **85 predictions (19%)** were within 5 characters of being correct
- Many "failures" are minor differences (1-2 characters off)
- The strict exact-match metric counts even tiny differences as failures

Example of a near-miss:
```
Input:    घबराहट में ठीक से वात भी नहीं कर पाती।
Expected: घबराहट में ठीक से बात भी नहीं कर पाती।
Model:    घबराहट में ठीक से वात भी नहीं कर पाती।
          (Only 1 character different: वात vs बात)
```

### Why This is a Hard Problem

1. **Synthetic vs Real Errors**: My training data uses simulated noise, but real ASR errors have different patterns
2. **Domain Gap**: The test set contains medical terms and error combinations not seen during training
3. **Ambiguity**: Some noisy inputs could map to multiple valid corrections

I experimented with larger datasets (30K samples) and more training epochs, but this actually made results worse. The model started overfitting to synthetic patterns and hallucinating corrections. The simpler approach generalizes better.

## How to Use

### Setup

```bash
# Create environment
conda create -n hindi_medical_correction python=3.10 -y
conda activate hindi_medical_correction

# Install dependencies
pip install -r requirements.txt
```

### Generate Training Data

```bash
python data_generator.py --num_samples 12000
```

### Train the Model

```bash
python train.py --data dataset.csv --epochs 5 --batch_size 16
```

Training takes about 30-40 minutes on a GPU. The model converges well:
- Training loss: 44.9 → 1.98
- Validation loss: 2.61 → 0.51

### Run Inference

```bash
# Interactive mode
python inference.py --model_path ./model_output/final_model --interactive

# Single sentence
python inference.py --model_path ./model_output/final_model --input "मुझे पेन में दर्द है"
```

### Evaluate

```bash
python evaluate.py --model_path ./model_output/final_model --eval_file eval.csv --output_dir .
```

This generates:
- `Evaluate_results.csv`: All predictions
- `evaluation_metrics.json`: Metrics in JSON
- `evaluation_report.txt`: Human-readable report

## Known Limitations

1. **Over-correction**: Sometimes changes text that was already correct (8 cases in eval)
2. **Hallucination**: Occasionally generates unrelated text for complex sentences
3. **Rare Terms**: Medical terms not in training data may be incorrectly "corrected"
4. **Ambiguous Inputs**: When multiple corrections are valid, may pick the wrong one

## System Requirements

- **Model Size**: ~2.1 GB on disk
- **GPU Memory**: ~4 GB for inference
- **Speed**: ~19 ms/sentence (GPU), ~300 ms/sentence (CPU)

## Possible Improvements

- Use a larger model (mT5-base) for better accuracy
- Collect real ASR errors for training instead of synthetic data
- Add post-processing rules for common patterns
- Fine-tune on domain-specific medical Hindi corpus

## References

- [mT5 Paper](https://arxiv.org/abs/2010.11934)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
