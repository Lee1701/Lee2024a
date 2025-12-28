#!/bin/bash
#SBATCH -p ycga
#SBATCH --qos=ycga_admins
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=300G
#SBATCH -t 2-
#SBATCH --job-name d3
#SBATCH --output maybridge.predictions.by.D3.models.%A.%a.log
#SBATCH --mail-user ho-joon.lee@yale.edu
#SBATCH --mail-type ALL
#SBATCH --array=0-127

echo "##### START #####"
date
echo

pwd
echo

ml miniconda
conda activate eddy

echo $CONDA_PREFIX
python -V
echo

export BATCH_SIZE=400
python -u maybridge.predictions.by.D3.models.py $SLURM_ARRAY_TASK_ID

conda deactivate

echo
date
echo "##### END #####"
