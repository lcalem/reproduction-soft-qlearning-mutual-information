#!/usr/bin/env bash

#SBATCH -J "grotile"
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --workdir=/homedtic/stotaro/baselines/
#SBATCH --mail-user="simone.totaro@gmail.com"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

set -x
#module load Python
#source ~/grotile/bin/activate

declare -a seeds=(10 50 100 400)
#module load Tensorflow
#module load pip
for seed in "${seeds[@]}"; do
#    sbatch run_envs.sh $envs $seed
    python main.py --agent $1 --env $2 --seed "$seed"
    break
done
