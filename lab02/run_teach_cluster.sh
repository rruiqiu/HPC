#!/bin/bash

##################### SLURM (do not change) v #####################
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --job-name="lab02"
#SBATCH --nodes=1
#SBATCH --output="lab02.%j.%N.out"
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

#TODO: Change the perf counters if needed.

if [[ $EXP == "spmm" ]]; then
$BINPATH/lab2_spmm  --benchmark_perf_counters="cache-misses,cache-references" --benchmark_format=csv --benchmark_out_format=csv  --benchmark_out=$LOGS/lab02_spmm.csv
elif [[ $EXP == "spmv" ]]; then
$BINPATH/lab2_spmv  --benchmark_perf_counters="cache-misses,cache-references" --benchmark_format=csv --benchmark_out_format=csv  --benchmark_out=$LOGS/lab02_spmv.csv
elif [[ $EXP == "spnn" ]]; then
$BINPATH/lab2_sparseNN  --benchmark_format=csv --benchmark_out_format=csv  --benchmark_out=$LOGS/lab02_spnn.csv
fi
  # plotting
  # You can add your plotting script here if not going beyond the time limit