#!/bin/bash

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Configuration
CHECKPOINT="lightning_logs/version_4/checkpoints/epoch=0-step=95-val_acc_epoch=0.0288.ckpt" # Update with your best checkpoint
DATA_FOLDER="data/RoBERTa-SST-5"
BATCH_SIZE=16
OUTPUT_DIR="results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run inference
python model/infer_model.py \
    --ckpt $CHECKPOINT \
    --concept_map ${DATA_FOLDER}/concept_idx.json \
    --batch_size $BATCH_SIZE \
    --paths_output_loc ${OUTPUT_DIR}/predictions.csv \
    --dev_file ${DATA_FOLDER}/dev_with_parse.json
