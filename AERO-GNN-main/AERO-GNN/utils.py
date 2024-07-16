
import random
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, WikiCS, WebKB, WikipediaNetwork, Actor
from torch_geometric.utils import remove_self_loops
from filtered_datasets import Filtered_Dataset as filtered_dataset
import pdb



def load_graph(args):
    
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        graph = Planetoid(root='./graph-data', name=args.dataset.capitalize(), split = 'public')[0]
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        transform = T.Compose([T.AddSelfLoops(), T.NormalizeFeatures()])
        graph = transform(graph)
    
    elif args.dataset == 'wiki':
        graph = WikiCS(root='./graph-data/Wiki-CS', is_undirected=True)[0]
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        transform = T.Compose([T.AddSelfLoops(), T.ToUndirected(),])
        graph = transform(graph)

    elif args.dataset in ['photo', 'computers']:
        graph = Amazon(root='./graph-data', name=args.dataset.capitalize())[0]
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        transform = T.Compose([T.AddSelfLoops(), T.NormalizeFeatures()])
        graph = transform(graph)
    
    elif args.dataset in ['texas', 'cornell', 'wisconsin']:
        graph = WebKB(root='./graph-data', name=args.dataset.capitalize())[0]
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        transform = T.Compose([T.AddSelfLoops(), T.ToUndirected(), T.NormalizeFeatures()])
        graph = transform(graph)

    elif args.dataset in ['squirrel', 'chameleon']:
        graph = WikipediaNetwork(root='./graph-data', name=args.dataset.capitalize())[0]
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        transform = T.Compose([T.AddSelfLoops(), T.ToUndirected(),])
        graph = transform(graph)

    elif args.dataset == 'actor':
        graph = Actor(root='./graph-data/actor')[0]
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        transform = T.Compose([T.AddSelfLoops(), T.ToUndirected(),])
        graph = transform(graph)

    elif args.dataset in ['chameleon-filtered', 'squirrel-filtered']:
        graph = filtered_dataset(name=args.dataset).pyg_graph
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        transform = T.Compose([T.AddSelfLoops(), T.ToUndirected(),])
        graph = transform(graph)

    return graph



def fixed_split(args, graph, exp_num):

    num_nodes = graph.x.size(0)

    if 'train_mask' in graph:
        if len(graph.train_mask.shape) > 1:
            num_splits = graph.train_mask.shape[1]
            split = exp_num % num_splits

            if args.dataset == 'wiki':
                train_idx = torch.nonzero(graph.train_mask[:,split]).squeeze()
                test_idx = torch.nonzero(graph.test_mask).squeeze()
                val_idx = torch.nonzero(graph.val_mask[:,split]).squeeze()

            else:
                train_idx = torch.nonzero(graph.train_mask[:,split]).squeeze()
                test_idx = torch.nonzero(graph.test_mask[:,split]).squeeze()
                val_idx = torch.nonzero(graph.val_mask[:,split]).squeeze()

        else:
            train_idx = torch.nonzero(graph.train_mask).squeeze()
            test_idx = torch.nonzero(graph.test_mask).squeeze()
            val_idx = torch.nonzero(graph.val_mask).squeeze()
            
    else:
        print('No train_mask found.')
        train_idx, val_idx, test_idx=random_splits(graph)
        

    return train_idx, val_idx, test_idx
def random_splits(graph, ratio=(0.6, 0.2, 0.2)):
        labels = torch.clone(graph.y)
        train_ratio, val_ratio, test_ratio = ratio
        assert sum(ratio) == 1.0
        num_classes= int(torch.max(labels).item() + 1)

        indices = []
        for i in range(num_classes):
            index = (labels == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append([index, len(index)])

        train_index = torch.cat([i[:int(train_ratio * num)] for i, num in indices], dim=0)
        val_index = torch.cat([i[int(train_ratio * num): int((train_ratio + val_ratio) * num)] for i, num in indices],
                              dim=0)
        test_index = torch.cat([i[int((train_ratio + val_ratio) * num):] for i, num in indices], dim=0)
      
        return train_index, val_index, test_index

def sparse_split(graph, train_ratio, val_ratio):

    num_nodes = graph.x.size(0)
    num_labels = int(graph.y.max() + 1)

    nodes = torch.arange(num_nodes)
    nodes = nodes[torch.randperm(num_nodes)]

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    train_idx = torch.LongTensor(nodes[0 : num_train])
    val_idx = torch.LongTensor(nodes[num_train : num_train+num_val])
    test_idx = torch.LongTensor(nodes[num_train+num_val : ])

    all_idx = torch.cat([train_idx, test_idx, val_idx])
    all_idx = torch.sort(all_idx)[0]
    assert torch.equal(all_idx, torch.arange(num_nodes))

    return train_idx, val_idx, test_idx


def init_optimizer(args, model):

    if args.model == 'aero':
        optimizer = torch.optim.Adam([
            {'params': model.dense_lins.parameters(),
            'weight_decay': args.dr},
            {'params': model.atts.parameters(),
            'weight_decay': args.dr_prop},
            {'params': model.hop_atts.parameters(),
            'weight_decay': args.dr_prop},
            {'params': model.hop_biases.parameters(),
            'weight_decay': args.dr_prop},
        ], lr=args.lr)
    
    
    elif args.model == 'gprgnn':
        optimizer = torch.optim.Adam([
            {'params': model.linear_node_1.parameters(),
            'weight_decay': args.dr},
            {'params': model.linear_node_2.parameters(),
            'weight_decay': args.dr},
            {'params': model.temp,
            'weight_decay': 0.0}
        ], lr=args.lr)

    elif args.model == 'gcn2':
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(),
            'weight_decay': args.dr},
            {'params': model.convs.parameters(),
            'weight_decay': args.dr_prop},
        ], lr=args.lr)

    elif args.model == 'adgn':
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(),
            'weight_decay': args.dr},
            {'params': model.conv.parameters(),
            'weight_decay': args.dr_prop},
        ], lr=args.lr)

    else:
        optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.dr)

    return optimizer
