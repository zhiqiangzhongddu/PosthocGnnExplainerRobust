import os.path as osp
import numpy as np
import torch

from torch_geometric.datasets import TUDataset, BA2MotifDataset, BAMultiShapesDataset
from torch_geometric.transforms import OneHotDegree
from dig.xgraph.dataset import SentiGraphDataset, MoleculeDataset


def load_data(args):
    path = './data'

    if args.dataset.lower() == "Mutagenicity".lower():
        dataset = TUDataset(root=path, name='Mutagenicity')

    elif args.dataset.lower() == "NCI1".lower():
        dataset = TUDataset(root=path, name='NCI1')

    elif args.dataset.lower() == "COLLAB".lower():
        dataset = TUDataset(root=path, name='COLLAB', transform=OneHotDegree(max_degree=491))

    elif args.dataset.lower() == "Synthie".lower():
        dataset = TUDataset(root=path, name='Synthie', transform=OneHotDegree(max_degree=20))

    elif args.dataset.lower() == "Graph_Twitter".lower():
        dataset = SentiGraphDataset(root=path, name="Graph-Twitter")

    elif args.dataset.lower() == "BA_2motifs".lower():
        dataset = BA2MotifDataset(root=path)

    elif args.dataset.lower() == "BAMultiShapes".lower():
        dataset = BAMultiShapesDataset(root=path)

    elif args.dataset.lower() == "ClinTox".lower():
        dataset = MoleculeDataset(root='data', name='clintox')
        dataset.data.y = torch.nonzero(dataset.data.y > 0)[:, 1]
        dataset.data.x = dataset.data.x.float()

    else:
        raise ValueError("%s is not a valid dataset option." % args.dataset)

    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of node features: {dim_node}')
    print(f'Number of edge features: {dim_edge}')
    print(f'Number of classes: {num_classes}')

    train_index, val_index, test_index = get_split(args=args, dataset=dataset)

    # split the dataset into train, validation and test set
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    print("N. train: %d, N. valid: %d, N. test: %d" % (len(train_dataset), len(val_dataset), len(test_dataset)))

    return dataset, train_dataset, val_dataset, test_dataset


def get_split(args, dataset):
    num_graph = len(dataset)
    train_path = "data/splits/%s_train.npy" % args.dataset
    val_path = "data/splits/%s_val.npy" % args.dataset
    test_path = "data/splits/%s_test.npy" % args.dataset

    if osp.exists(train_path):
        train_index = np.load(train_path)
        val_index = np.load(val_path)
        test_index = np.load(test_path)
        assert train_index.shape[0] + val_index.shape[0] + test_index.shape[0] == num_graph

    else:
        print("generating splits for %s" % args.dataset)
        indices = np.arange(num_graph)
        np.random.shuffle(indices)

        train_index = indices[: int(num_graph * 0.7)]
        val_index = indices[int(num_graph * 0.7): int(num_graph * 0.8)]
        test_index = indices[int(num_graph * 0.8):]
        # train_index = indices[: int(num_graph * 0.8)]
        # val_index = indices[int(num_graph * 0.8): int(num_graph * 0.9)]
        # test_index = indices[int(num_graph * 0.9):]
        np.save(train_path, train_index)
        np.save(val_path, val_index)
        np.save(test_path, test_index)

    return train_index, val_index, test_index
