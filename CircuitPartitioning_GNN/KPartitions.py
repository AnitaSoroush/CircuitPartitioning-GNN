import os
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
import os
import sys
import time
from torch.nn import Linear
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GraphConv, dense_mincut_pool
from torch_geometric import utils
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.metrics import normalized_mutual_info_score as NMI
from torch_geometric.utils import degree
import networkx as nx
import numpy as np
import torch.nn as nn


# Functions
from cirpart.utils import get_fp, Netlist

# Constants
from cirpart.globals import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_features(num_nodes, edge_index):
    # Convert edge_index to NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())
    # print(G)

    # Compute node degrees
    node_degrees = degree(edge_index[0], num_nodes=num_nodes)

    # Compute node centrality
    centrality = list(nx.degree_centrality(G).values())
    # print("Centrality values:", centrality)

    # Compute average neighbor degree
    avg_neighbor_degree = list(nx.average_neighbor_degree(G).values())
    # print("Centrality values:", avg_neighbor_degree)

    # Compute clustering coefficient
    cluster_coeff = list(nx.clustering(G).values())

    return np.array([node_degrees, centrality, avg_neighbor_degree, cluster_coeff]).T



def process(netlist, num_classes = 2):
    edge_index=torch.tensor(netlist.edge_index)
    edge_attr=torch.tensor(netlist.edge_weights)
    
    node_features = torch.tensor(compute_features(netlist.Nmods, edge_index), dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=None, num_classes = num_classes)
    return data , netlist

def netlist_info(netlist_name):
    netlist = Netlist()
    netlist.load_dat(get_fp(netlist_name))
    return netlist


# the following part calulates the number of cutsize based on a given assignment for a given netlist
def cut_size_calculator(netlist, pred):
    pred_dict = {}
    for i, assignment in enumerate(pred.tolist()): pred_dict[i+1] = assignment
    netlist.group_dict = pred_dict
    return netlist.calculate_cutsize()

class CostumeLoss(nn.Module):
    def __init__(self):
        super(CostumeLoss, self).__init__()

    def forward(self, embeddings, edge_index, edge_weight, normalized_A_sparse, D_sparse):    
        # Compute S^T * A * S
        term1 = torch.sparse.mm(torch.sparse.mm(embeddings.t(), normalized_A_sparse), embeddings).trace()    
        
        # Compute S^T * D * S
        term2 = torch.sparse.mm(torch.sparse.mm(embeddings.t(), D_sparse), embeddings).trace()
        
        # Compute the loss
        loss = -(term1/term2)
        
        # ---------------penalty ----------------------------
        # probabilities: tensor of shape (num_nodes, num_clusters) containing probabilities
        # k: number of clusters
        
        # Ensure that probabilities require gradients
        k = embeddings.shape[1]
        
        # Compute Prob^T * Prob
        prob_transpose_prob = torch.matmul(embeddings.t(), embeddings)
        # print(prob_transpose_prob)
        
        # Compute the Frobenius norm of Prob^T * Prob
        norm_prob_transpose_prob = torch.norm(prob_transpose_prob, p='fro')
        
        # Compute the normalized Prob^T * Prob
        normalized_prob_transpose_prob = prob_transpose_prob / norm_prob_transpose_prob
        
        # Compute identity matrix I
        identity = torch.eye(k)
        
        # Compute the Frobenius norm of the difference between normalized Prob^T * Prob and I
        penalty = torch.norm(normalized_prob_transpose_prob - (identity / torch.sqrt(torch.tensor(k, dtype=torch.float32))), p='fro')
        return loss + penalty 


class Net(torch.nn.Module):
    def __init__(self, 
                 mp_units,
                 mp_act,
                 in_channels, 
                 n_clusters, 
                 mlp_units=[]):
        super().__init__()
        
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        
        # Message passing layers
        mp = [
            # (GraphConv(in_channels, mp_units[0]), 'x, edge_index, edge_weight -> x'),
            (GCNConv(in_channels, mp_units[0]), 'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            # mp.append((GraphConv(mp_units[i], mp_units[i+1]), 'x, edge_index, edge_weight -> x'))
            mp.append((GCNConv(mp_units[i], mp_units[i+1]), 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]
        
        # MLP layers
        self.mlp = torch.nn.Sequential()
        self.mlp.append(Linear(out_chan, n_clusters))
        

    def forward(self, x, edge_index, edge_weight):
        
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight) 
        
        # Cluster assignments (logits)
        s = self.mlp(x) 
        
        s = F.softmax(s, dim=-1)
        return s
    

# INPUT EXAMPLE: python KPartitions.py ibm01 2

if __name__ == '__main__':
    netlist_name = sys.argv[1]
    num_classes = int(sys.argv[2])

    set_seed(7)
    results = []
    logs = []
    netlist = netlist_info(netlist_name)
    number_of_modules, number_of_nets = netlist.Nmods, netlist.Nnets
    print (netlist_name)
    
    result_filename = "results/" + netlist_name + ".pickle"
    print (result_filename)
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    
    data, netlist = process(netlist, num_classes = num_classes)
    data.edge_index, data.edge_weight = gcn_norm(  
            data.edge_index, data.edge_weight, data.num_nodes,
            add_self_loops=False, dtype=data.x.dtype)
    
    # Construct the sparse adjacency matrix
    A_sparse = torch.sparse_coo_tensor(data.edge_index, data.edge_weight, (number_of_modules, number_of_modules))
    
    # Normalize the adjacency matrix (row-normalization)
    row_sum = torch.sparse.sum(A_sparse, dim=1).to_dense().squeeze()
    D_sqrt_inv = torch.sqrt(1.0 / row_sum)
    D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0  # Handle division by zero
    D_sqrt_inv = torch.sparse_coo_tensor(torch.stack([torch.arange(number_of_modules), torch.arange(number_of_modules)]), D_sqrt_inv, (number_of_modules, number_of_modules))
    normalized_A_sparse = torch.sparse.mm(D_sqrt_inv, torch.sparse.mm(A_sparse, D_sqrt_inv))  
    
    # Compute the degree matrix
    degrees = torch.sparse.sum(normalized_A_sparse, dim=1).to_dense().squeeze()  # Sum along rows to get degree of each node
    D_sparse = torch.sparse_coo_tensor(torch.stack([torch.arange(number_of_modules), torch.arange(number_of_modules)]), degrees, (number_of_modules, number_of_modules))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = Net([16, 32, 32], "ELU", data.num_features, data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    costume_loss = CostumeLoss()

    general_time = 0
    for epoch in range(1, 100):
        begin = time.time()
        model.train()
        optimizer.zero_grad()
        s = model(data.x, data.edge_index, data.edge_weight)
        loss = costume_loss(s, data.edge_index, data.edge_weight, normalized_A_sparse, D_sparse)
        loss.backward()
        # print('Gradients:')
        # for param in model.parameters():
        #     print(param.grad)
        optimizer.step()
        end = time.time()
        train_loss = loss.item()
        general_time += (end - begin)
        # print(loss)
        
        _, class_counts = np.unique(s.argmax(dim=1), return_counts=True)
      
        # saving logs:
        log = {
            "Loss": train_loss,
            "netcut": cut_size_calculator(netlist, s.argmax(dim=1)),
            "netcut ratio" : (cut_size_calculator(netlist, s.argmax(dim=1))/number_of_nets),
            "time": end-begin
        } 
        logs.append(log)
        
        
    model.eval()
    s = model(data.x, data.edge_index, data.edge_weight)
    pred = s.argmax(dim=1)
    import numpy as np
    _, class_counts = np.unique(pred, return_counts=True)
    print(_)
    print(class_counts)
    
    cut_size = cut_size_calculator(netlist, pred)
    
    res = {"Netlist Name:": netlist_name, 
            "# of modolues": number_of_modules,
            "# of nets": number_of_nets,
            "net cuts":cut_size,
            'net cut ratio': (cut_size/number_of_nets),
            "time": (general_time/200)}
    
    results.append(res)

    print("name:", netlist_name)
    print('number of nets:{}'.format(number_of_nets))
    print('number of net cuts: {}'.format(cut_size))
    print('time: {:.3f}'.format(general_time))

    # writing the results
    with open(result_filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(s, handle, protocol=pickle.HIGHEST_PROTOCOL)