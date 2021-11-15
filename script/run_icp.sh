#!/bin/bash

ROOT=$(pwd)

# path to dataset
DATA_DIR=$ROOT/data/2D/v1_pose0
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=icp_v1_pose0
CKPT_DIR=$ROOT/results
# Error metrics for ICP
# point: "point2point"
# plane: "point2plane"
METRIC=plane

python $ROOT/deepmapping/incremental_icp.py --name $NAME -d $DATA_DIR -m $METRIC -c $CKPT_DIR
