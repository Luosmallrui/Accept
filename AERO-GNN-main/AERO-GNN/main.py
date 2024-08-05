# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:47:41 2024
@author: wangranr
"""
from Train import Trainer as s_trainer
import torch
from utils import load_graph
import argparse
import pdb
#torch.autograd.set_detect_anomaly(True)
def parameter_parser():
    parser = argparse.ArgumentParser()
    """DATASET"""
    parser.add_argument("--dataset", type=str, default="cora",)
    """EXPERIMENT"""
    parser.add_argument("--exp-num", type=int, default=10)
    parser.add_argument("--model", type=str, default="gatv2")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, )
    parser.add_argument("--device", nargs="?", default="cuda:0", )#cuda:0
    parser.add_argument("--split", type=str, default="fixed", )
    parser.add_argument("--task", type=str, default="node-cls", )
    parser.add_argument("--epochs", type=int, default=1000, )
    parser.add_argument("--layers", type=int, default= 2, )
    parser.add_argument("--walk-length", type=int, default=100, )
    parser.add_argument("--walks-per-node", type=int, default=6, )
    parser.add_argument("--batch-size", type=int, default= 32, )
    """HYPERPARAMETERS"""   
    parser.add_argument("--iterations", type=int, default=1) # L_P
    parser.add_argument("--num-layers", type = int, default =1) # L_M
    parser.add_argument("--dropout", type=float, default=0.6,)
    parser.add_argument("--add-dropout", type = int, default = 0) # Last Layer Dropout
    parser.add_argument("--lr", type=float, default=0.001, )
    parser.add_argument("--dr", type=float, default=0.0005,) # WD_M
    parser.add_argument("--dr-prop", type=float, default=0.0005, ) # WD_P
    parser.add_argument("--hid-dim", type=int, default=64,) 
    parser.add_argument("--num-heads", type=int, default=8, ) # Attention Heads
    parser.add_argument("--lambd", type=float, default=0.01,) # Decay Weighting for GCNII and AERO-GNN
    parser.add_argument("--alpha", type=float, default=0.2, ) #  smoothing factor
    parser.add_argument("--lambd-l2", type=float, default=0,) # L2 Reg
    return parser.parse_args()
def main():
    # load dataset
    args = parameter_parser()
    graph = load_graph(args)
    # sparse labeled training
    if args.dataset in ['cora', 'citeseer', 'wiki', 'pubmed', 'photo', 'computers']:
        args.epochs  = 2000
        args.early_stopping_rounds = 100
        args.lr = 0.01
        s_trainer(args, graph).fit()
    # dense labeled training
    else: 
        args.epochs = 5000
        args.early_stopping_rounds = 500
        if args.model not in ['fagcn', 'gcn2']: args.lr = 0.005 # the two models tune their lr \in {0.005, 0.01}
        s_trainer(args, graph).fit()
if __name__ == "__main__":
    main()