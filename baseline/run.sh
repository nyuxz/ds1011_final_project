#!/bin/bash
#
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=CBOW
#SBATCH --mail-type=END
#SBATCH --mail-user=xz1757@nyu.edu
#SBATCH --output=slurm_%j.out

module load pytorch/python3.5/0.2.0_3

time python3 CBOW_MLP.py --num_labels 3 --hidden_dim 5
