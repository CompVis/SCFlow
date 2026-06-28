#!/bin/bash

CONFIG_PATH="configs/catfm_training.yaml"
EXPERIMENT_NAME="catfm_train"

python training.py \
  --name $EXPERIMENT_NAME \
  --config $CONFIG_PATH
