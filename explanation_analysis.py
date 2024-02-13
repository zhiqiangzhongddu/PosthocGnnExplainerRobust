from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_remaining_self_loops

from metrics import XCollector
from utils import kl_divergence, frechet_distance, avg_distance


# Define function to generate explanations and calculate KL divergence
def generate_explanations_and_kl_div_batch(dataset, indices, explainer_pyg, device, batch_size=64):
    kl_div_list = []

    data_loader = DataLoader([dataset[idx] for idx in indices], batch_size=batch_size, shuffle=False)

    for data_t in tqdm(data_loader):
        data_t.to(device)

        explanation_t = explainer_pyg(
            x=data_t.x,
            edge_index=data_t.edge_index,
            batch=data_t.batch,  # Batch of size 1
            target=data_t.y
        ).to(device)

        explanation_f = explainer_pyg(
            x=data_t.x,
            edge_index=data_t.edge_index,
            batch=data_t.batch,  # Batch of size 1
            target=1-data_t.y
        ).to(device)

        for idx, str in enumerate(data_t.ptr[:-1]):
            end = data_t.ptr[idx + 1]

            kl_div = kl_divergence(explanation_f['edge_mask'][str:end], explanation_t['edge_mask'][str:end])
            kl_div_list.append(kl_div.item())

    return kl_div_list


# Define function to plot
def plot_res(res_to_plot, group_name, matrix: str=""):
    df = pd.DataFrame({
        'Index': range(len(res_to_plot)),
        matrix: res_to_plot
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='Index', y=matrix)
    plt.title('%s for Each Index (%s)' % matrix, group_name)
    plt.show()


def scatterplot_res_together(res_lists, group_names, matrix: str=""):
    # List to store dataframes
    dfs = []

    for fd_list, group_name in zip(res_lists, group_names):
        df = pd.DataFrame({
            'Index': range(len(fd_list)),
            matrix: fd_list,
            'Group': [group_name] * len(fd_list)
        })
        dfs.append(df)

    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=final_df, x='Index', y=matrix, hue='Group', palette='tab10')
    plt.title('%s for Each Index by Group' % matrix)
    plt.show()


def lmplot_res_together(res_lists, group_names, matrix: str=""):
    # List to store dataframes
    dfs = []

    for fd_list, group_name in zip(res_lists, group_names):
        df = pd.DataFrame({
            'Index': range(len(fd_list)),
            matrix: fd_list,
            'Group': [group_name] * len(fd_list)
        })
        dfs.append(df)

    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.lmplot(data=final_df, x='Index', y=matrix, hue='Group', palette='tab10', ci=None)
    plt.title('%s for Each Index by Group' % matrix)
    plt.show()


def generate_explanations_and_dist(dataset, indices, explainer_pyg, matrix, device):
    list_res = []

    for idx in tqdm(indices):
        data_t = dataset[idx].to(device)
        data_f = data_t.clone()
        data_f.y = 1 - data_f.y

        explanation_t = explainer_pyg(
            x=data_t.x,
            edge_index=data_t.edge_index,
            batch=torch.tensor([0 for _ in range(data_t.x.shape[0])]).to(device),  # Batch of size 1
            target=data_t.y
        ).to(device)

        explanation_f = explainer_pyg(
            x=data_f.x,
            edge_index=data_f.edge_index,
            batch=torch.tensor([0 for _ in range(data_f.x.shape[0])]).to(device),  # Batch of size 1
            target=data_f.y
        ).to(device)

        edge_mask_dict_t = {}
        for i, edge_value in enumerate(explanation_t['edge_mask']):
            edge_mask_dict_t[(data_t.edge_index[0][i].item(), data_t.edge_index[1][i].item())] = edge_value.item()

        edge_mask_dict_f = {}
        for i, edge_value in enumerate(explanation_f['edge_mask']):
            edge_mask_dict_f[(data_f.edge_index[0][i].item(), data_f.edge_index[1][i].item())] = edge_value.item()

        edge_mask_t = torch.tensor([edge_mask_dict_t.get(edge, 0.0) for edge in edge_mask_dict_t])
        edge_mask_f = torch.tensor([edge_mask_dict_f.get(edge, 0.0) for edge in edge_mask_dict_f])

        if matrix == "kl_div":
            list_res.append(kl_divergence(edge_mask_f, edge_mask_t))
        elif matrix == "frechet_dist":
            list_res.append(frechet_distance(edge_mask_f, edge_mask_t))
        elif matrix == "avg_dist":
            list_res.append(avg_distance(edge_mask_f, edge_mask_t))
        else:
            ValueError("%s is a wrong matrix type" % matrix)

    return list_res


def explanations_dist(explanation_t, explanation_f, matrix):
    list_res = []

    for idx in tqdm(range(len(explanation_t))):

        edge_mask_t = explanation_t[idx]
        edge_mask_f = explanation_f[idx]

        if 0 in edge_mask_t:
            edge_mask_t = edge_mask_t[torch.nonzero(edge_mask_t != 0)].squeeze()
            edge_mask_f = edge_mask_f[torch.nonzero(edge_mask_f != 0)].squeeze()

        if matrix == "kl_div":
            list_res.append(kl_divergence(edge_mask_f, edge_mask_t))

        elif matrix == "frechet_dist":
            list_res.append(frechet_distance(edge_mask_f, edge_mask_t))

        elif matrix == "avg_dist":
            list_res.append(avg_distance(edge_mask_f, edge_mask_t))

        else:
            ValueError("%s is a wrong matrix type" % matrix)

    return list_res


def save_edge_mask_pyg(dataset, indices, explainer_pyg, data_name, gnn_name, explainer_name, indices_name, device):
    res_t = []
    res_f = []
    for idx in tqdm(indices):
        data_t = dataset[idx].to(device)
        data_f = data_t.clone()
        data_f.y = 1 - data_f.y

        explanation_t = explainer_pyg(
            x=data_t.x,
            edge_index=data_t.edge_index,
            batch=torch.tensor([0 for _ in range(data_t.x.shape[0])]).to(device),  # Batch of size 1
            target=data_t.y
        ).to(device)

        explanation_f = explainer_pyg(
            x=data_f.x,
            edge_index=data_f.edge_index,
            batch=torch.tensor([0 for _ in range(data_f.x.shape[0])]).to(device),  # Batch of size 1
            target=data_f.y
        ).to(device)

        res_t.append(explanation_t.edge_mask)
        res_f.append(explanation_f.edge_mask)

    # pad all tensors to have same length
    max_len = max([x.squeeze().numel() for x in res_t])
    res_t = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in res_t]
    res_f = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in res_f]

    res_t = torch.stack(res_t)
    res_f = torch.stack(res_f)
    torch.save(res_t, "./edge_masks/%s_%s_%s_edge_mask_%s_t.pth" % (data_name, gnn_name, explainer_name, indices_name))
    torch.save(res_f, "./edge_masks/%s_%s_%s_edge_mask_%s_f.pth" % (data_name, gnn_name, explainer_name, indices_name))


def generate_explanation_dig(args, model, explainer, dataset, device):
    # --- Create data collector and explanation processor ---
    ex_list = [] if args.save_explanation else None

    if args.explainer_model in ["GNNExplainer", "RobustGNNExplainer"]:
        # --- Set the Sparsity to 0.5 ---
        sparsity = 0.5
        x_collector = XCollector(sparsity=sparsity)

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, data in enumerate(tqdm(data_loader)):
            data.to(device)

            edge_masks, hard_edge_masks, related_preds = explainer(
                x=data.x, edge_index=data.edge_index, sparsity=sparsity, num_classes=dataset.num_classes,
            )
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
            x_collector.collect_data(masks=hard_edge_masks, related_preds=related_preds, label=pred.long().item())
            if args.save_explanation:
                ex_list.append(edge_masks[pred.long().item()])

    elif args.explainer_model in ["PGExplainer", "RobustPGExplainer"]:
        x_collector = XCollector()

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataloader_tqdm = tqdm(data_loader, "Generate explanation for each data")
        for i, data in enumerate(dataloader_tqdm):
            data.to(device)
            if data.edge_index.shape[1] == 0:
                data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)

            with torch.no_grad():
                walks, masks, related_preds = explainer(
                    data.x, data.edge_index
                )
                masks = [mask.detach() for mask in masks]
            x_collector.collect_data(masks=masks, related_preds=related_preds)
            if args.save_explanation:
                ex_list.append(masks[0])

    else:
        raise ValueError("%s is not a available explainer model" % args.explainer_model)

    return x_collector, ex_list


def save_edge_masks_dig(args, ex_list):

    # pad all tensors to have same length
    max_len = max([ex.squeeze().numel() for ex in ex_list])
    ex_list = [torch.nn.functional.pad(ex, pad=(0, max_len - ex.numel()), mode='constant', value=0) for ex in ex_list]
    ex_tensor = torch.stack(ex_list)
    save_file = "./edge_masks/%s_%s_%s_edge_mask_%s.pth" % (
        args.dataset, args.gnn_model, args.explainer_model, int(args.manipulate_ratio * 100)
    )
    print("saving edge_masks at %s" % save_file)
    torch.save(ex_tensor, save_file)
