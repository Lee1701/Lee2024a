#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -t 1-
#SBATCH --job-name d1
#SBATCH --output maybridge.predictions.by.D1.models.%A.%a.log
#SBATCH --mail-user ho-joon.lee@yale.edu
#SBATCH --mail-type ALL
#SBATCH --array=0-127

echo "##### START #####"
date
echo

pwd
echo

ml anaconda3
conda activate eddy01

echo $CONDA_PREFIX
python -V
echo

export SHARED=/ocean/projects/cis240078p/shared
export LOCAL=/local

export BATCH_SIZE=400
python -u maybridge.predictions.by.D1.models.py $SLURM_ARRAY_TASK_ID

conda deactivate

echo
date
echo "##### END #####"
