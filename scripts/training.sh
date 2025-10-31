#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"


CONFIG_PATH="configs/training.yaml"
EXPERIMENT_NAME="default"
USE_WANDB=false
USE_WANDB_OFFLINE=false
RESUME_CKPT=""
LOAD_WEIGHTS=""
NUM_NODES=1
DEVICES=-1
FIND_UNUSED_PARAMS=false


python training.py \
  --config $CONFIG_PATH \
  --name $EXPERIMENT_NAME \
  $( [ "$USE_WANDB" = true ] && echo "--use_wandb" ) \
  $( [ "$USE_WANDB_OFFLINE" = true ] && echo "--use_wandb_offline" ) \
  $( [ -n "$RESUME_CKPT" ] && echo "--resume_checkpoint $RESUME_CKPT" ) \
  $( [ -n "$LOAD_WEIGHTS" ] && echo "--load_weights $LOAD_WEIGHTS" ) \
  --num_nodes $NUM_NODES \
  --devices $DEVICES \
  $( [ "$FIND_UNUSED_PARAMS" = true ] && echo "--find_unused_parameters" )
