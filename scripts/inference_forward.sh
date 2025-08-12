#!/bin/bash
export CUDA_VISIBLE_DEVICES=6


CONFIG_PATH="configs/inference.yaml"
CHECKPOINT_PATH="ckpts/scflow_last.ckpt"
OUTPUT_FOLDER_NAME="vis_output"
IMAGE_C_PATH="image_samples/Cubism/02316.png"
IMAGE_S_PATH="image_samples/Drip_Painting/09728.png"
UNCLIP_CHECKPOINT_PATH="ckpts/sd21-unclip-l.ckpt"
SEED=2025

python inference.py \
    --config "$CONFIG_PATH" \
    --resume_checkpoint "$CHECKPOINT_PATH" \
    --name "$OUTPUT_FOLDER_NAME" \
    --image_c_path "$IMAGE_C_PATH" \
    --image_s_path "$IMAGE_S_PATH" \
    --unclip_ckpt "$UNCLIP_CHECKPOINT_PATH" \
    --seed $SEED