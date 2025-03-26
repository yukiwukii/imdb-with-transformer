#!/bin/sh
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=ollama
#SBATCH --nodelist=TC1N05
#SBATCH --time=06:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama list
python llamaIMDB.py
