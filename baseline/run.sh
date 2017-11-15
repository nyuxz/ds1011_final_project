#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -t24:00:00
#SBATCH --job-name=CBOW
#SBATCH --mail-type=END
#SBATCH --mail-user=qc510@nyu.edu
#SBATCH --output=slurm_%j.out

module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3

module load cuda/8.0.44
module load cudnn/8.0v5.1

time python3 CBOW_MLP.py
