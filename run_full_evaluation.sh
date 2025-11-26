#!/bin/bash
set -e

echo "=== FACE VERIFICATION FULL EVALUATION PIPELINE ==="
echo "Starting full evaluation pipeline..."
echo ""

echo "=== GENERATING NEGATIVE PAIRS ==="
echo "Generating similarity pairs for threshold tuning..."
python src/generate_pairs.py

echo ""
echo "=== THRESHOLD TUNING ==="
echo "Running threshold optimization analysis..."
python src/threshold_tuning.py \
  --pos data/evaluation/similarities_pos.npy \
  --neg data/evaluation/similarities_neg.npy \
  --out_dir data/evaluation/threshold

echo ""
echo "=== EVALUATING QUERY IMAGES ==="
echo "Evaluating query images against PersonA embeddings..."
python src/evaluate_queries.py \
  --query_dir data/processed/query_images \
  --emb_path data/embeddings/personA_normalized.npz

echo ""
echo "=== DONE ==="
echo "Full evaluation pipeline completed successfully!"
echo "Results saved in:"
echo "  - data/evaluation/ (query evaluation results)"
echo "  - data/evaluation/threshold/ (threshold optimization results)"
echo ""