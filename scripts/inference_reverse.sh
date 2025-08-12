#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

CONFIG_PATH="configs/inference.yaml"
CHECKPOINT_PATH="ckpts/scflow_last.ckpt"
OUTPUT_FOLDER_NAME="vis_output"
REVERSE_INFERENCE=true
IMAGE_MIX_PATH="image_samples/Cyberpunk/02316.png"
UNCLIP_CHECKPOINT_PATH="ckpts/sd21-unclip-l.ckpt"
SEED=2025


python inference.py \
    --config "$CONFIG_PATH" \
    --resume_checkpoint "$CHECKPOINT_PATH" \
    --name "$OUTPUT_FOLDER_NAME" \
    --reverse_inference "$REVERSE_INFERENCE" \
    --image_mix_path "$IMAGE_MIX_PATH" \
    --unclip_ckpt "$UNCLIP_CHECKPOINT_PATH" \
    --seed $SEED