#!/bin/bash
MODEL="Qwen3.5-9B"

echo "Running RZ Category..."
python rz_category.py --model $MODEL

echo "Running RZ Section..."
python rz_section.py --model $MODEL

echo "Running RZ Clinical..."
python rz_clinical.py --model $MODEL

echo "Running RZ Observability..."
python rz_observability.py --model $MODEL

echo "Running RZ Full..."
python rz_full.py --model $MODEL