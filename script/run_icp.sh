#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=deepmapping_icp
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=narendasan@nyu.edu
#SBATCH --output=deepmapping_icp_%j.stdout

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
