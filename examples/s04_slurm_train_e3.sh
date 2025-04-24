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

FULLSET=${1:-"./data/sample-regression"}
TRAINRATIO=${2:-"0.6"}
VALIDRATIO=${3:-"0.2"}
TESTRATIO=${4:-"0.2"}

srun --unbuffered cgcnne3-tr \
	--full-set $FULLSET \
	--train-ratio $TRAINRATIO \
	--valid-ratio $VALIDRATIO \
	--test-ratio $TESTRATIO \
	--epoch 1e3 \
	--stop-patience 1e2 \
	--learning-rate 1e-2 \
	--job-id $SLURM_JOB_ID \
	--random-seed 42

mv stdout.${SLURM_JOB_ID} $OUTDIR
mv stderr.${SLURM_JOB_ID} $OUTDIR

exit 0
