#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=deepmapping_ma_training
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=narendasan@nyu.edu
#SBATCH --output=deepmapping_ma_training_%j.stdout


ROOT=$(pwd)

# Enviornment to use
MAP=v1
# Number of agents in the enviornment
NUM_AGENTS=5
# Location of agent obvservations
DATA=$ROOT/data/multi_agent
# Location to store results
RESULTS=$ROOT/results/multi_agent
# Number of local samples
N=19
# Number of training epochs for each agent
EPOCHS=3000
# Batch size
BS=128
# Logging intervals
LOG=20
# Learning rate
LR=0.001

python3 $ROOT/deepmapping/mutli_agent_deepmapping.py  \
	--env $MAP \
	--num-agents $NUM_AGENTS \
	--data-dir $DATA \
	--results-dir $RESULTS \
  --local-lr $LR \
	--local-samples $N \
	--local-epochs $EPOCHS \
	--local-batch-size $BS \
	--save-interval $LOG \
	--restart
