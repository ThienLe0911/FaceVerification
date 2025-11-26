# Evaluation Data

This directory contains similarity data generated for threshold tuning and ROC analysis.

## Generated Files

### Core Output Files
- **`similarities_pos.npy`** - Positive pair similarities (PersonA images vs PersonA mean embedding)
- **`similarities_neg.npy`** - Negative pair similarities (Others images vs PersonA mean embedding)  
- **`neg_filenames.npy`** - Filenames of negative images successfully processed
- **`metadata.json`** - Generation metadata and statistics

### Metadata Content
```json
{
  "timestamp": "2025-11-23T...",
  "n_images_pos": 27,
  "n_images_neg": 1000, 
  "embedding_dim": 512,
  "seed": 42,
  "emb_path": "data/embeddings/personA_normalized.npz",
  "others_dir": "data/processed/others",
  "requested_neg": 1000,
  "actual_neg": 950
}
```

## Usage

### 1. Generate Similarity Pairs
```bash
# Generate with default settings (use all negatives)
python src/generate_pairs.py

# Generate with 1000 random negative samples
python src/generate_pairs.py --num_neg 1000 --seed 42

# Custom paths
python src/generate_pairs.py \
    --emb_path data/embeddings/personA_normalized.npz \
    --others_dir data/processed/others \
    --out_dir data/evaluation \
    --num_neg 500
```

### 2. Load Results for Analysis
```python
import numpy as np
import json

# Load similarities
pos_sim = np.load('data/evaluation/similarities_pos.npy')
neg_sim = np.load('data/evaluation/similarities_neg.npy') 
neg_files = np.load('data/evaluation/neg_filenames.npy')

# Load metadata
with open('data/evaluation/metadata.json') as f:
    meta = json.load(f)

print(f"Positive pairs: {len(pos_sim)}")
print(f"Negative pairs: {len(neg_sim)}")
print(f"Pos mean: {pos_sim.mean():.3f}")
print(f"Neg mean: {neg_sim.mean():.3f}")
```

## Analysis Pipeline

### Next Steps (DAY 7 - Timeline)
1. **Threshold Analysis**: Use similarities for ROC curve generation
2. **EER Calculation**: Find Equal Error Rate
3. **Optimal Threshold**: Select threshold based on target FPR/TPR
4. **Visualization**: Plot histogram of positive vs negative similarities

### Expected Similarity Ranges
- **Positive similarities**: ~0.85-0.99 (high similarity)
- **Negative similarities**: ~0.20-0.60 (low similarity)
- **Good separation**: Positive min > Negative max

## File Sizes (Approximate)
- similarities_pos.npy: ~216 bytes (27 floats)
- similarities_neg.npy: ~4KB-8KB (1000-2000 floats)
- neg_filenames.npy: ~20KB-40KB (1000-2000 strings)
- metadata.json: ~500 bytes

## Troubleshooting

### Common Issues
1. **No face detected**: Some images in others/ may not have detectable faces
2. **Memory usage**: Large numbers of negatives may use significant RAM
3. **Processing time**: Embedding generation for negatives can take 5-30 minutes

### Performance Tips
- Use `--num_neg` to limit negative samples for faster processing
- Monitor progress via INFO logs every 100 images
- Results are deterministic with same `--seed` value