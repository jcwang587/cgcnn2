#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --job-name=cgcnn2
#SBATCH --output stdout.%j
#SBATCH --error stderr.%j
#SBATCH --partition=gpu

#######################################################################
ulimit -s unlimited
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate cgcnn2-env
#######################################################################

OUTDIR="output_${SLURM_JOB_ID}"
mkdir -p $OUTDIR

cp $0 $OUTDIR

MODELPATH=${1:-"../cgcnn2/pretrained_models/formation-energy-per-atom.pth.tar"}
FULLSET=${2:-"./data/sample-regression"}
TRAINRATIO=${3:-"0.6"}
VALIDRATIO=${4:-"0.2"}
TESTRATIO=${5:-"0.2"}

srun --unbuffered cgcnn-ft \
	--model-path $MODELPATH \
	--full-set $FULLSET \
	--train-ratio $TRAINRATIO \
	--valid-ratio $VALIDRATIO \
	--test-ratio $TESTRATIO \
	--epoch 1e3 \
	--stop-patience 1e2 \
	--lr-fc 0.01 \
	--lr-non-fc 0.001 \
	--reset \
	--train-last-fc \
	--job-id $SLURM_JOB_ID \
	--random-seed 42

mv stdout.${SLURM_JOB_ID} $OUTDIR
mv stderr.${SLURM_JOB_ID} $OUTDIR

exit 0
