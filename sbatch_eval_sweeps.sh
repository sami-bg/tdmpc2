#!/bin/bash
#SBATCH -o /users/sboughan/reai/tdmpc2/logs/sweep_evals-%j.out
#SBATCH -e /users/sboughan/reai/tdmpc2/logs/sweep_evals-%j.err
#SBATCH --job-name=sweep_evals
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus=1

# Default values
task=""
checkpoints=()
aggregations=(mean)
horizon_eval=(3 5 7 10 15)
var_coeff=(0.0 0.001 0.01 1.0 10.0)
seeds=(1 2 3)

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            task="$2"
            shift 2
            ;;
        --checkpoints)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                checkpoints+=("$1")
                shift
            done
            ;;
        --aggregations)
            aggregations=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                aggregations+=("$1")
                shift
            done
            ;;
        --horizon_eval)
            horizon_eval=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                horizon_eval+=("$1")
                shift
            done
            ;;
        --var_coeff)
            var_coeff=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                var_coeff+=("$1")
                shift
            done
            ;;
        --seeds)
            seeds=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                seeds+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$task" ]; then
    echo "Error: --task is required"
    exit 1
fi

if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "Error: --checkpoints is required"
    exit 1
fi

conda activate tdmpc2
cd tdmpc2

# Run evaluation for each combination
for checkpoint in "${checkpoints[@]}"; do
    for h in "${horizon_eval[@]}"; do
        for agg in "${aggregations[@]}"; do
            for var in "${var_coeff[@]}"; do
                for seed in "${seeds[@]}"; do
                    python evaluate.py \
                        task=$task \
                        checkpoint=$checkpoint \
                        horizon_eval=$h \
                        ensemble_aggregation=$agg \
                        var_coeff=$var \
                        seed=$seed
                done
            done
        done
    done
done