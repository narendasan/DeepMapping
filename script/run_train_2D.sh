#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=deepmapping_training
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=narendasan@nyu.edu
#SBATCH --output=deepmapping_training_%j.stdout


ROOT=$(pwd)

# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=v1_pose0
# path to dataset
DATA_DIR=$ROOT/data/2D/v1_pose0
# training epochs
EPOCH=3000
# batch size
BS=128
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=19
# logging interval
LOG=20

CKPT_DIR=$ROOT/results

### training from scratch
#python train_2D.py --name $NAME -d $DATA_DIR -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
INIT_POSE=$ROOT/results/2D/icp_v1_pose0/pose_est.npy
python $ROOT/deepmapping/train_2D.py --name $NAME -d $DATA_DIR -i $INIT_POSE -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG -c $CKPT_DIR
