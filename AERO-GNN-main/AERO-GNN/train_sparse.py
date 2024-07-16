import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
import pdb
from tqdm import trange
import os
import csv
from models import *
from utils import fixed_split, sparse_split, init_optimizer

control = {
    "dagnn": "iterations",
    "mixhop": "num_layers",
    "appnp": "iterations",
    "adgn" :  "iterations",
    "fagcn" : "iterations",
    "graphsage": "num_layers",
}


class Trainer(object):

    def __init__(self, args, graph):
        self.test_accuracy = 0
        self.args = args
        self.graph = graph
        self.target = self.graph.y

        self.device = torch.device(self.args.device)
        torch.cuda.set_device(self.device)

        self.in_channels = self.graph.x.size(1)
        self.hid_channels = self.args.hid_dim
        self.out_channels = int(torch.max(self.target).item() + 1)

    def create_model(self):
        model_mapping = {
            'aero': AERO_GNN_Model,
            'gcn': GCN_Model,
            'appnp': APPNP_Model,
            'gcn2': GCNII_Model,
            'adgn': ADGN_Model,
            'gat': GAT_Model,
            'gatv2': GAT_v2_Model,
            'gt': GT_Model,
            'gat-res': GAT_v2_Res_Model,
            'fagcn': FAGCN_Model,
            'gprgnn': GPR_GNN_Model,
            'dagnn': DAGNN_Model,
            'mixhop': MixHop_Model,
            'graphsage': GraphSAGE
        }
        # choose model
        Model = model_mapping.get(self.args.model)
        self.model = Model(self.args,
                           self.in_channels,
                           self.hid_channels,
                           self.out_channels,
                           self.graph,
                           )

    def data_split(self):

        """
        train/val/test split
        """
        if self.args.split == 'fixed': split = fixed_split(self.args, self.graph, self.exp)
        if self.args.split == 'sparse': split = sparse_split(self.graph, 0.025, 0.025)
        self.train_nodes, self.validation_nodes, self.test_nodes = split

    def transfer_to_gpu(self):

        if self.exp == 0: self.graph = self.graph.to(self.device)
        self.target = self.target.long().squeeze().to(self.device)
        self.model = self.model.to(self.device)

    def calculate_dirichlet_energy(self, X, edge_index):
        """
        Calculate the Dirichlet energy for a given feature matrix X and edge index.
        
        Parameters:
        X (torch.Tensor): Feature matrix of shape (num_nodes, num_features)
        edge_index (torch.Tensor): Edge index matrix of shape (2, num_edges)
        
        Returns:
        torch.Tensor: Dirichlet energy
        """
        num_nodes = X.size(0)

        # Calculate node degrees
        degrees = torch.zeros(num_nodes, dtype=X.dtype, device=X.device)
        degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=X.dtype, device=X.device))

        # Normalize features by sqrt(1 + degrees)
        norm_factors = torch.sqrt(1 + degrees).unsqueeze(1)
        X_norm = X / norm_factors

        # Compute differences between connected nodes
        diff = X_norm[edge_index[0]] - X_norm[edge_index[1]]

        # l2 norm
        dirichlet_energy = torch.norm(diff, p=2, dim=1).sum() / num_nodes
        return dirichlet_energy / 2

    def eval(self, index_set):

        self.model.eval()

        with torch.no_grad():
            X, prediction = self.model(self.graph.x, self.graph.edge_index)
            logits = F.log_softmax(prediction, dim=1)
            loss = F.nll_loss(logits[index_set], self.target[index_set])

            _, pred = logits.max(dim=1)
            correct = pred[index_set].eq(self.target[index_set]).sum().item()
            acc = correct / len(index_set)
            X = X.view(prediction.size(0), -1)
            dirichlet_energy = self.calculate_dirichlet_energy(X, self.graph.edge_index)

            return acc, loss, dirichlet_energy

    def do_a_step(self):

        self.model.train()
        self.optimizer.zero_grad()
        _, prediction = self.model(self.graph.x, self.graph.edge_index)
        prediction = F.log_softmax(prediction, dim=1)
        self.loss = F.nll_loss(prediction[self.train_nodes], self.target[self.train_nodes])

        if self.args.lambd_l2 > 0:
            self.loss += sum([p.pow(2).sum() for p in self.model.parameters()]) * self.args.lambd_l2

        self.loss.backward()
        self.optimizer.step()

    def train_neural_network(self):

        self.optimizer = init_optimizer(self.args, self.model)
        self.iterator = trange(self.args.epochs, desc='Validation accuracy: ', leave=False)

        self.step_counter = 0
        self.best_val_acc = 0
        self.best_val_loss = np.inf

        for _ in self.iterator:

            self.do_a_step()

            val_acc, val_loss, val_dirichlet_energy = self.eval(self.validation_nodes)
            self.iterator.set_description("Validation accuracy: {:.4f}".format(val_acc))

            close = self.loss_step_counter(val_loss)
            if close: break

    def loss_step_counter(self, val_loss):
        # self.test_dirichlet_energy=0
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.test_accuracy, _, self.test_dirichlet_energy = self.eval(self.test_nodes)
            self.step_counter = 0
            return False

        else:
            self.step_counter = self.step_counter + 1
            if self.step_counter > self.args.early_stopping_rounds:
                self.iterator.close()
                return True
            return False

    def fit(self):

        acc = []
        dirichlet_energies = []
        seeds = torch.load('../seeds_100.pt')

        for _ in range(self.args.exp_num):
            self.exp = _
            self.seed = seeds[_]

            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            self.model = None
            self.optimizer = None

            self.create_model()
            self.data_split()
            self.transfer_to_gpu()
            self.train_neural_network()

            acc.append(self.test_accuracy)
            dirichlet_energies.append(self.test_dirichlet_energy)
            print("Trial {:} Test Accuracy: {:.4f}, Dirichlet Energy: {:.4f}".format(self.exp, self.test_accuracy,
                                                                                     self.test_dirichlet_energy))

        self.avg_acc = sum(acc) / len(acc)
        self.std_acc = torch.std(torch.tensor(acc)).item()
        self.avg_dirichlet_energy = sum(dirichlet_energies) / len(dirichlet_energies)
        self.std_dirichlet_energy = torch.std(torch.tensor(dirichlet_energies, dtype=torch.float32)).item()

        c = control[self.args.model]
        if c == "iterations":
            layer = self.args.iterations
        else:
            layer = self.args.num_layers
        print(f"layer:{layer}")
        print("epoch", self.args.epochs)
        print("Model: {}".format(self.args.model))
        print('n trials: {}'.format(self.args.exp_num))
        print('dataset: {}'.format(self.args.dataset))
        print("Mean test accuracy: {:.4f}".format(self.avg_acc), "±", '{:.3f}'.format(self.std_acc))
        print("Mean Dirichlet energy: {:.4f}".format(self.avg_dirichlet_energy), "±",
              '{:.3f}'.format(self.std_dirichlet_energy))
        epoch = self.args.epochs
        model = self.args.model
        n_trials = self.args.exp_num
        dataset = self.args.dataset
        avg_acc = self.avg_acc
        std_acc = self.std_acc
        avg_dirichlet_energy = self.avg_dirichlet_energy
        std_dirichlet_energy = self.std_dirichlet_energy

        csv_file = os.path.join('../result', f'{self.args.model}.csv')
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(
                    ['layer', 'epoch', 'Model', 'n trials', 'dataset', 'Mean test accuracy', 'std deviation',
                     'Mean Dirichlet energy', 'std Dirichlet energy'])
            writer.writerow([layer, epoch, model, n_trials, dataset, f"{avg_acc:.4f}", f"{std_acc:.3f}",
                             f"{avg_dirichlet_energy:.4f}", f"{std_dirichlet_energy:.3f}"])
