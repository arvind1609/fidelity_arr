#!/bin/bash
MODEL="Qwen3.5-9B"

echo "Running EV Category..."
python ev_category.py --model $MODEL

echo "Running EV Section..."
python ev_section.py --model $MODEL

echo "Running EV Clinical..."
python ev_clinical.py --model $MODEL

echo "Running EV Observability..."
python ev_observability.py --model $MODEL

echo "Running EV Full..."
python ev_full.py --model $MODEL
