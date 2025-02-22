#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# python -c "from transformers import AutoModel; AutoModel.from_pretrained('xlnet-base-cased', force_download=True)"
# Run for XLNet
# python model/run.py --dataset_basedir data/XLNet-SUBJ \
#     --lr 2e-5 --max_epochs 5 \
#     --gpus 1 \
#     --concept_store data/XLNet-SUBJ/concept_store.pt \
#     --accelerator ddp \
#     --gamma 0.1 \
#     --lamda 0.1 \
#     --topk 5

# for RoBERTa
# python model/run.py --dataset_basedir data/RoBERTa-SUBJ \
#                          --lr 2e-5  --max_epochs 5 \
#                          --gpus 1 \
#                          --concept_store data/RoBERTa-SUBJ/concept_store.pt \
#                          --accelerator ddp \
#                          --model_name roberta-base \
#                          --topk 5 \
#                          --gamma 0.1 \
#                          --lamda 0.1

# python -c "from transformers import AutoModel; AutoModel.from_pretrained('roberta-base', force_download=True)"
# Run for RoBERTa-SST-5

#epoch should be at least 5
python run.py --dataset_basedir data/RoBERTa-SST-5 \
    --lr 2e-5 --max_epochs 1 \
    --concept_store data/RoBERTa-SST-5/concept_store.pt \
    --accelerator ddp \
    --gamma 0.1 \
    --lamda 0.1 \
    --topk 5 \
    --model_name roberta-base
