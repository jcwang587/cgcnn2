#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --job-name=cgcnn
#SBATCH --output stdout.%j
#SBATCH --error stderr.%j
#SBATCH --partition=gpu

#######################################################################
ulimit -s unlimited
module add miniconda/22.11.1-1
eval "$(conda shell.bash hook)"
conda activate llto-kmc 
#######################################################################

# Create output directory
OUTDIR="output_${SLURM_JOB_ID}"
mkdir -p $OUTDIR

cp $0 $OUTDIR

MODE=${1:-"1"}
MODELPATH=${2:-"./gnn_model/gen0/struct/struct_model.ckpt"}
TOTALSET=${3:-"./data/train_struct_cif"}
TRAINRATIO=${4:-"0.6"}
VALIDRATIO=${5:-"0.2"}
TESTRATIO=${6:-"0.2"}

srun --unbuffered python ../bin/cgcnn_ft.py \
	--mode $MODE \
	--model-path $MODELPATH \
	--total-set $TOTALSET \
	--train-ratio $TRAINRATIO \
	--valid-ratio $VALIDRATIO \
	--test-ratio $TESTRATIO \
	--epoch 1e7 \
	--lr-fc 0.01 \
	--lr-non-fc 0.001 \
	--train-last-fc 0 \
	--stop-patience 2e4 \
	--lr-patience 0 \
	--lr-factor 0.0 \
	--replace 1 \
	--bias-temperature 0.0 \
	--job-id $SLURM_JOB_ID \
	--random-seed 411 \

# Move stdout and stderr to output directory
mv stdout.${SLURM_JOB_ID} $OUTDIR
mv stderr.${SLURM_JOB_ID} $OUTDIR

exit 0
