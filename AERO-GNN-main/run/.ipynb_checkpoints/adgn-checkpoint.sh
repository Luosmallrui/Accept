#!/bin/bash

# Define arrays for datasets and iterations
datasets=( "cora")
iterations=( 64)

# Loop through each dataset and each iteration value
for dataset in "${datasets[@]}"; do
  for iter in "${iterations[@]}"; do
    python ../AERO-GNN/main.py --model adgn --dataset "$dataset" --iterations "$iter"
  done
done
