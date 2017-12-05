#!/bin/bash
#
#SBATCH --cpus-per-task=2
#SBATCH --time=40:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=mgru_att
#SBATCH --mail-type=END
#SBATCH --mail-user= email
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

module load pytorch/python3.5/0.2.0_3

module load cuda/8.0.44
module load cudnn/8.0v5.1

time python3 mgru_att.py 