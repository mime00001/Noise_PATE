#!/bin/bash
#---SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=5G
#SBATCH -c 4
#SBATCH --ntasks 1
# SBATCH --nodelist=sprint1
#SBATCH --output=/home/michel/project/Noise_PATE/output/%j.out

# mitigates activation problems
eval "$(conda shell.bash hook)"
source .bashrc

# activate the correct environment
# conda activate pytorch
# source activate /sprint1/anaconda/envs/pytorch
# source activate /home/${USER}/.conda/envs/test-env
conda activate noise-pate

# debug print-outs
echo USER: $USER
which conda
which python

# run the code
PYTHONPATH=. python dpsgd_script.py
#python mnist.py --backbone_name "shaders21k_grey" --delta 10e-5 --epochs 10 --data-root "/storage3/michel/data/"