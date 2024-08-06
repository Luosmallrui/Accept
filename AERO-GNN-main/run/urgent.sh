#!/bin/bash

# Define arrays for datasets and iterations
datasets=("computers" )
iterations=( 1 2 4 8 16 32 64)

# # Loop through each dataset and each iteration value
# for dataset in "${datasets[@]}"; do
#   for iter in "${iterations[@]}"; do
#     python ../AERO-GNN/main.py --model gat --dataset "$dataset" --num-layers "$iter"  --device cuda:0 
#   done
# done

python ../AERO-GNN/main.py --model gt --dataset "computers" --num-layers 64  --device cuda:0 --dr 1e-3 --dropout 0.6 --lambd-l2 0.0
python ../AERO-GNN/main.py --model gat-res --dataset "computers" --num-layers 64  --device cuda:0 --dr 1e-3 --dropout 0.6 --lambd-l2 0.0