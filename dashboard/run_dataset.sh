#!/bin/bash
source .env
DATA_DIR=$ROOT_DIR/$1
PORT=$2
echo "Data dir: $DATA_DIR"
echo "Port: $PORT"
[ ! -d '$DATA_DIR' ] && mkdir '$DATA_DIR'
eval "$(conda shell.bash hook)"
conda activate rt-cetsa-dash
DATA_DIR=$DATA_DIR PORT=$PORT solara run src/rt_cetsa/dashboard.py