#!/bin/sh
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=finetune
#SBATCH --time=06:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

conda init
conda activate nndl
python finetune/bart_finetune.py
