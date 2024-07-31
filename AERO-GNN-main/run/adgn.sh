#!/bin/bash

# Define arrays for datasets and iterations
datasets=("cora" "citeseer" "pubmed" )
iterations=( 1 2 4 8 16 32 64)

# Loop through each dataset and each iteration value
for dataset in "${datasets[@]}"; do
  for iter in "${iterations[@]}"; do
    python ../AERO-GNN/main.py --model adgn --dataset "$dataset" --iterations "$iter" --device cuda:1
  done
done
