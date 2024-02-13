import os
import numpy as np
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx


def parse_args():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--fast_mode", help="if use fast mode", type=str2bool, default=False)
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--gpu_id", help="device", type=str, default="0")
    parser.add_argument("--save_explanation", help="if save explanation", type=str2bool, default=False)

    # dataset
    parser.add_argument("--dataset", help="dataset name", type=str, default="BA_2motifs",
                        choices=["Mutagenicity", "NCI1", "BA_2motifs", "BAMultiShapes", "ClinTox", "Synthie", "Graph_Twitter"])
    parser.add_argument("--manipulate_ratio", help="manipulate ratio", type=float, default=0.)

    # GNN
    parser.add_argument("--gnn_model", help="model architecture", type=str, default="GCN_3l",
                        choices=['GCN_2l', 'GCN_2l_BN', 'GCN_3l', 'GCN_3l_BN', 'GIN_2l', 'GIN_2l_BN','GIN_3l', 'GIN_3l_BN'])
    parser.add_argument("--gnn_hid_dim", help="hidden dimension of GNN encoder", type=int, default=128)

    # Explainer
    parser.add_argument("--explainer_model", help="explainer method", type=str, default="GNNExplainer",
                        choices=['GNNExplainer', 'PGExplainer'])

    # execution
    parser.add_argument("--gnn_epoch", help="number of epochs for GNN", type=int, default=200)
    parser.add_argument("--gnn_lr", help="learning rate for GNN", type=float, default=0.001)
    parser.add_argument("--ex_epoch", help="number of epochs for explainer", type=int, default=100)
    parser.add_argument("--ex_lr", help="learning rate for explainer", type=float, default=0.001)

    return parser.parse_known_args()[0]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed: int = 42):

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return


# Save the model in a file
def save_model(model, filename):
    torch.save(model.state_dict(), filename)


# Load the model from a file
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    os.remove(filename)


def to_molecule(data):
    ATOM_MAP = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        data['name'] = ATOM_MAP[data['x'].index(1.0)]
        del data['x']
    return g


def contains_NO2(data):

    molecule = to_molecule(data)

    for node, data in molecule.nodes(data=True):
        if data['name'] == 'N':
            # Get neighbors
            neighbors = list(molecule.neighbors(node))
            if len(neighbors) < 2:
                continue
            # Check if Nitrogen is connected to two Oxygens
            oxygen_count = sum(1 for neighbor in neighbors if molecule.nodes[neighbor]['name'] == 'O')
            if oxygen_count >= 2:
                return True
    return False


def count_NO2(data):

    molecule = to_molecule(data)

    count = 0
    for node, data in molecule.nodes(data=True):
        if data['name'] == 'N':
            # Get neighbors
            neighbors = list(molecule.neighbors(node))
            # Check if Nitrogen is connected to two Oxygens
            oxygen_count = sum(1 for neighbor in neighbors if molecule.nodes[neighbor]['name'] == 'O')
            if oxygen_count >= 2:
                count += 1
    return count


# Define function for KL Divergence calculation
def kl_divergence(edge_mask1, edge_mask2):
    edge_mask1_normalized = F.normalize(edge_mask1, p=1, dim=0)
    edge_mask2_normalized = F.normalize(edge_mask2, p=1, dim=0)

    kl_div = torch.sum(
        edge_mask1_normalized * (torch.log(edge_mask1_normalized + 1e-9) - torch.log(edge_mask2_normalized + 1e-9))
    )

    return float(kl_div)
# def kl_divergence(p, q):
#     return float(torch.sum(p * torch.log(p / q), -1))


def frechet_distance(edge_mask1, edge_mask2):

    dist = (edge_mask1 - edge_mask2).abs().max()

    return float(dist)


def avg_distance(edge_mask1, edge_mask2):

    dist = (edge_mask1 - edge_mask2).abs().sum() / edge_mask1.size(0)

    return float(dist)


def compute_distance(edge_mask1, edge_mask2, matrix):
    dist = 0

    if "kl" in matrix.lower():
        dist += kl_divergence(edge_mask1, edge_mask2)

    if 'frechet' in matrix.lower() or 'our' in matrix.lower():
        dist += frechet_distance(edge_mask1, edge_mask2)

    if 'avg' in matrix.lower() or 'our' in matrix.lower():
        dist += avg_distance(edge_mask1, edge_mask2)

    return dist


def manipulate_dataset(dataset, ratio: float):

    _indices = np.array(dataset._indices)

    new_y = dataset.data.y.clone()

    for label in range(dataset.num_classes):
        to_replace_label_set = list(set(range(dataset.num_classes)) - {label})
        indices = torch.nonzero(dataset.y == label).squeeze().numpy().tolist()
        selected_index = random.sample(indices, int(len(indices) * ratio))
        new_y[_indices[selected_index]] = torch.tensor(np.random.choice(to_replace_label_set, len(selected_index)))

    dataset.data.y = new_y

    return dataset


def visualise_molecule(dataset, explanation, n_id, plotutils, figname=None):
    # n_id = 35
    # explanation = explanation_0

    # plot molecular graph
    data = dataset[n_id].cpu()
    edge_mask = explanation[n_id][torch.nonzero(explanation[n_id] != 0)].squeeze().cpu()
    print("Number of NO2: ", count_NO2(data))

    graph = to_networkx(data)
    plotutils.plot_soft_edge_mask(
        graph=graph,
        edge_mask=edge_mask[:data.num_edges],
        top_k=5,
        un_directed=True,
        figname=figname,
        x=data.x,
    )


def visualise_distance(dataset, explanation, explanation_f, matrix, manipulation):
    # explanation = explanation_0
    # explanation_f = explanation_25
    # matrix = "Our"

    dist = []
    num_no2 = []
    for n_id in range(len(dataset)):
        # count number of NO2
        data = dataset[n_id].cpu()
        num_no2.append(count_NO2(data))
        edge_mask = explanation[n_id][torch.nonzero(explanation[n_id] != 0)].squeeze().cpu()
        edge_mask_f = explanation_f[n_id][torch.nonzero(explanation_f[n_id] != 0)].squeeze().cpu()
        dist.append(compute_distance(edge_mask1=edge_mask, edge_mask2=edge_mask_f, matrix=matrix))

    df = pd.DataFrame({
        'Index': range(len(dataset)),
        'Distance': dist,
        'No. NO2': num_no2
    })

    df = df[df['No. NO2'] < 3]

    # Plotting
    fig = sns.lmplot(
        data=df, x='Index', y='Distance', hue='No. NO2', palette='tab10', ci=None, legend_out=False,
        scatter_kws={"s": 20}, height=5, aspect=1.5
    )
    if manipulation == 0:
        plt.title('%s distance by group - w/o' % matrix)
    elif manipulation == 25:
        plt.title('%s distance by group - w/.25' % matrix)
    elif manipulation == 50:
        plt.title('%s distance by group - w/.5' % matrix)
    elif manipulation == 100:
        plt.title('%s distance by group - w/1.' % matrix)

    plt.show()

    return fig


# class GumbelSoftmaxTransform:
#     def __init__(self, tau=1.0):
#         self.tau = tau
#
#     def _sample_gumbel(self, shape, eps=1e-20):
#         U = torch.rand(shape)
#         return -torch.log(-torch.log(U + eps) + eps)
#
#     def transform(self, probabilities):
#
#         gumbel_noise = self._sample_gumbel(probabilities.size())
#         log_probs = torch.log(probabilities) + gumbel_noise
#         softmax_probs = F.softmax(log_probs / self.tau, dim=-1)
#
#         return softmax_probs


# import torch
# import matplotlib.pyplot as plt
#
# def zoom_function(x, a=20, b=0.5):
#     return 1 / (1 + torch.exp(-a * (x - b)))
#
# # input
# input_values = torch.linspace(0, 1, 100)
#
# # change parameter
# a_values = [5, 10, 20]
# b_value = 0.5  # 固定 b 参数
#
# # plot with different gradients
# plt.figure(figsize=(8, 6))
# for a in a_values:
#     output_values = zoom_function(input_values, a, b_value)
#     plt.plot(input_values, output_values, label=f'a = {a}')
#
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.title('Custom Function Similar to Sigmoid')
# plt.legend()
# plt.grid(True)
# plt.show()
def zoom_function(x, a=20, b=0.5):
    return 1 / (1 + torch.exp(-a * (x - b)))