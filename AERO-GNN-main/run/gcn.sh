#!/bin/bash

# Define arrays for datasets and iterations
datasets=("cora" "wiki" "pubmed" "computers" "actor")
iterations=(1 2 4 8 16 32 64)

# Loop through each dataset and each iteration value
for dataset in "${datasets[@]}"; do
  # Set default parameters
  dr="0.001"
  dropout="0.8"
  lambd_l2="0.0"
  split=""

  # Override parameters based on dataset
  case $dataset in
    "cora")
      dr="0.001"
      dropout="0.8"
      lambd_l2="0.0"
      ;;
    "wiki")
      dr="0.0001"
      dropout="0.6"
      lambd_l2="0.0"
      ;;
    "pubmed")
      dr="0.001"
      dropout="0.7"
      lambd_l2="0.0"
      ;;
    "computers")
      dr="0.0001"
      dropout="0.5"
      split="sparse"
      ;;
    "actor")
      dr="0.001"
      dropout="0.7"
      lambd_l2="0.0005"
      ;;
    # Add more cases for other datasets as needed
    *)
      echo "Unknown dataset: $dataset"
      continue
      ;;
  esac

  # Loop through each iteration for the current dataset
  for iter in "${iterations[@]}"; do
    python ../AERO-GNN/main.py --model gcn --dataset "$dataset" --num-layers "$iter" --dr "$dr" --dropout "$dropout" --lambd-l2 "$lambd_l2" --split "$split" 
  done
done
