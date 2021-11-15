#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=deepmapping_eval
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=narendasan@nyu.edu
#SBATCH --output=deepmapping_eval_%j.stdout

ROOT=$(pwd)
CHECKPOINT_DIR=$ROOT/results/2D/v1_pose0/
python eval_2D.py -c $CHECKPOINT_DIR
