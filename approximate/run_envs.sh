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

declare -a agents=("sql" "dqn" "gl" "mirl")
declare -a envs="utils/env.txt"
declare -a gym_env="MiniGrid-FourRooms-v0"


#while IFS='' read -r line || [[ -n "$line" ]]
#do
for agent in "${agents[@]}"; do
    echo "$agent started with env $line"
    sbatch run_experiment.sh "$agent" "$gym_env"
    break
done
#        sbatch main.py --env "$line" --agent "$agent" --seed "$2"
#        sbatch run_experiment.sh "$agent" "$line"
#        sleep 5
#break
#done <$envs
