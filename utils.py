import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd

from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
import episcanpy.api as epi
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from termcolor import colored


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    # print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized), sparse_mx_to_torch_sparse_tensor(adj_)


def construct_spatial_graph(data, n_neighbors=5):

    # Select one omics to construct the spatial graph (all omics share the same spatial structure)
    any_key = list(data.keys())[0]  # Select any omics (e.g., "RNA")
    adata_ref = data[any_key].copy()

    # Construct spatial adjacency matrix based on spatial coordinates
    cell_positions = adata_ref.obsm["spatial"]
    adj_spatial = construct_graph_by_coordinate(cell_positions, n_neighbors=n_neighbors)

    # Transform adjacency matrix to sparse format
    adj_spatial = transform_adjacent_matrix(adj_spatial)
    adj_spatial = adj_spatial.toarray()  # Ensure symmetry
    adj_spatial = adj_spatial + adj_spatial.T
    adj_spatial = np.where(adj_spatial > 1, 1, adj_spatial)

    # Normalize adjacency matrix
    adj_processed, adj_label = preprocess_graph(adj_spatial)

    # Convert to PyTorch tensor and move to device
    adj_dense = {
        "adj_spatial": adj_processed.to_dense(),
        "adj_spatial_label": adj_label.to_dense()
    }

    return adj_dense


def print_metrics(ACC, F1, NMI, ARI, AMI, VMS, FMS):
    metrics_str = f"| ACC: {ACC:.4f} | NMI: {NMI:.4f} | ARI: {ARI:.4f} | F1: {F1:.4f} | AMI: {AMI:.4f} | VMS: {VMS:.4f} | FMS: {FMS:.4f} |"
    border = "=" * len(metrics_str)

    colored_border = colored(border, 'white', attrs=['bold'])
    colored_metrics = colored(metrics_str, 'red', attrs=['bold'])

    print(colored_border)
    print(colored_metrics)
    print(colored_border)

from sklearn.cluster import KMeans

def hellinger_kmeans(X, n_clusters, random_state=None):
    X_normalized = X / X.sum(axis=1, keepdims=True)
    X_sqrt = np.sqrt(X_normalized)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(X_sqrt)
    centroids_sqrt = kmeans.cluster_centers_
    centroids = centroids_sqrt ** 2
    centroids = centroids / centroids.sum(axis=1, keepdims=True)
    return kmeans.labels_, centroids

def fix_seed(seed):
    # seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print('Set the random seed to {}'.format(seed))