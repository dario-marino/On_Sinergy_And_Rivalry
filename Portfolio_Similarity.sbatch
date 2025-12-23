#!/bin/bash
#SBATCH --job-name=Portfolio_Similarity
#SBATCH --output=Portfolio_Similarity.out
#SBATCH --error=Portfolio_Similarity.err
#SBATCH --account=pi-jevans
#SBATCH --partition=amd-hm
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=0

# Load Python module
module load python/anaconda-2022.05

python /home/dariomarino/Downloads/Portfolio_Similarity.py