#!/bin/bash

##################### SLURM (do not change) v #####################
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --job-name="lab03"
#SBATCH --nodes=1
#SBATCH --output="lab03.%j.%N.out"
#SBATCH -t 00:25:00
##################### SLURM (do not change) ^ #####################

module load TeachEnv/2022a
module load gcc/13.2.0
module load cmake

### do not change above


# TODO add your binary files here to run them in the cluster


BINPATH=./build/
LOGS="./build/logs/"
#LOGS="./build/logs-${DATE}/"

EXP=$1

mkdir $LOGS

#TODO: Change/ADD the perf counters if needed.

if [[ $EXP == "mm" ]]; then
$BINPATH/lab3_mm  --benchmark_format=csv --benchmark_out_format=csv  --benchmark_out=$LOGS/lab03_mm.csv
elif [[ $EXP == "integration" ]]; then
$BINPATH/lab3_integration  --benchmark_format=csv --benchmark_out_format=csv  --benchmark_out=$LOGS/lab03_integration.csv
fi
  # plotting
  # You can add your plotting script here if not going beyond the time limit