#!/bin/bash
MODEL="Qwen3.5-9B"
STEPS=1

echo "Running Reflection Category..."
python reflection_category.py --model $MODEL --steps $STEPS

echo "Running Reflection Section..."
python reflection_section.py --model $MODEL --steps $STEPS

echo "Running Reflection Full..."
python reflection_full.py --model $MODEL --steps $STEPS

echo "Running Reflection Observability..."
python reflection_observability.py --model $MODEL --steps $STEPS

echo "Running Reflection Clinical..."
python reflection_clinical.py --model $MODEL --steps $STEPS