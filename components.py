import torch.autograd
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .ECR import ECR


# Graph convolutional layers
class G_layer(nn.Module):
    def __init__(self, num_sample: int, hidden_dim: int, activation: callable = F.softplus):
        super(G_layer, self).__init__()

        # Initialize weights with Xavier uniform distribution
        self.weight = nn.Parameter(torch.empty(num_sample, hidden_dim))
        torch.nn.init.xavier_uniform_(self.weight)

        self.activation = activation

    def forward(self, inputs: torch.Tensor, adj: torch.Tensor, activate: bool = False) -> torch.Tensor:

        x = inputs
        w = torch.mm(x, x.T)  # [num_sample, num_sample]
        x = torch.mul(w, adj)  # Hadamard product
        x = torch.mm(x, self.weight)
        if activate:
            x = self.activation(x)
        else:
            x = x
        return x


class GHN_encoder(nn.Module):
    def __init__(self, num_sample: int, num_topics: int, hidden_dim: int):
        super(GHN_encoder, self).__init__()
        self.f1 = G_layer(num_sample, hidden_dim)

        self.mu = G_layer(num_sample, num_topics)
        self.log_var = G_layer(num_sample, num_topics)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.f1(x, adj, activate=True)

        mu = self.mean_bn(self.mu(h, adj))
        log_var = self.logvar_bn(self.log_var(h, adj))
        return mu, log_var


# class GraphSAGE_Encoder(nn.Module):
#     """
#     Modality-specific GraphSAGE encoder.
#
#     Parameters
#     ----------
#     in_feat: int
#         Dimension of input features.
#     out_feat: int
#         Dimension of output features.
#     dropout: float
#         Dropout probability for latent representations.
#     act: Activation function. By default, we use ReLU.
#
#     Returns
#     -------
#     Latent representation.
#     """
#
#     def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
#         super(GraphSAGE_Encoder, self).__init__()
#         self.in_feat = in_feat
#         self.out_feat = out_feat
#         self.dropout = dropout
#         self.act = act
#
#         # Learnable weights
#         self.weight_self = nn.Parameter(torch.FloatTensor(self.in_feat, self.out_feat))  # Self-loop weights
#         self.weight_neigh = nn.Parameter(torch.FloatTensor(self.in_feat, self.out_feat))  # Neighbor weights
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # Initialize weights with Xavier uniform distribution
#         torch.nn.init.xavier_uniform_(self.weight_self)
#         torch.nn.init.xavier_uniform_(self.weight_neigh)
#
#     def forward(self, feat, adj):
#         # Compute neighbor aggregation: mean pooling
#         neigh_feat = torch.spmm(adj, feat)  # Sparse matrix multiplication for neighbor aggregation
#
#         # Apply learnable weights to self and neighbor features
#         self_feat = torch.mm(feat, self.weight_self)  # Transformation of self features
#         neigh_feat = torch.mm(neigh_feat, self.weight_neigh)  # Transformation of aggregated neighbor features
#
#         # Combine self and neighbor features
#         out = self_feat + neigh_feat
#
#         # Apply activation function
#         out = self.act(out)
#
#         # Apply dropout
#         out = F.dropout(out, p=self.dropout, training=self.training)
#
#         return out


class decoder(nn.Module):
    def __init__(self, omics_dim: int,
                 num_topics: int,
                 topic_embeddings: torch.Tensor,
                 weight_loss_ECR: float,
                 sinkhorn_alpha: float,
                 OT_max_iter: int,
                 beta_temp: float):
        super(decoder, self).__init__()

        assert topic_embeddings.requires_grad, "Topic embeddings should be learnable"

        self.num_topics = num_topics
        self.omics_dim = omics_dim
        self.embedding_dim = topic_embeddings.shape[1]
        self.topic_embeddings = topic_embeddings
        self.beta_temp = beta_temp

        self.omics_embeddings = nn.Parameter(
            F.normalize(
                nn.init.trunc_normal_(torch.empty(omics_dim, self.embedding_dim)),
                p=2, dim=1
            )
        )

        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, OT_max_iter)

        self.decoder_bn = nn.BatchNorm1d(omics_dim)
        self.decoder_bn.weight.requires_grad = False

        self.eps = 1e-9

    def get_beta(self) -> torch.Tensor:
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.omics_embeddings)
        beta = F.softmax(-dist / (self.beta_temp + self.eps), dim=0)

        return beta

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.omics_embeddings)
        loss_ECR = self.ECR(cost)

        return loss_ECR

    def forward(self, theta: torch.Tensor, omics: torch.Tensor) -> dict:
        omics_beta = self.get_beta()
        recon_omics = F.softmax(self.decoder_bn(torch.matmul(theta, omics_beta)), dim=-1)

        recon_loss = -(omics * recon_omics.log()).sum(axis=1).mean()
        loss_ECR = self.get_loss_ECR()

        rst_dict = {
            'recon_loss': recon_loss,
            'loss_ECR': loss_ECR,
            'recon_omics': recon_omics,
        }

        return rst_dict


class Decoder_graph(nn.Module):
    def __init__(self):
        super(Decoder_graph, self).__init__()

    def forward(self, eta: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        inner_product = torch.matmul(eta, eta.T)
        tnp = torch.sum(eta ** 2, dim=1).reshape(-1, 1).expand(size=inner_product.shape)
        A_pred = torch.sigmoid(- (tnp - 2 * inner_product + tnp.T) + a)
        return A_pred


class GNNLayer(nn.Module):
    """
    A single GNN layer that performs a linear transformation of the input
    features followed by matrix multiplication with the adjacency matrix (adj).
    """

    def __init__(self, in_features, out_features, activation=nn.Softplus()):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.act = activation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights using Xavier initialization.
        """
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        """
        Forward pass through the GNN layer.
        - features: Input node features.
        - adj: Adjacency matrix representing graph structure.
        - apply_activation: Whether to apply the activation function.
        """
        if active:
            support = self.act(torch.mm(features, self.weight))
        else:
            support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)

        return output


class GCN_encoder(nn.Module):
    def __init__(self, omics_dim: int, num_topics: int, hidden_dim: int):
        super(GCN_encoder, self).__init__()
        self.f1 = GNNLayer(omics_dim, hidden_dim)

        self.mu = GNNLayer(hidden_dim, num_topics)
        self.log_var = GNNLayer(hidden_dim, num_topics)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.f1(x, adj, active=True)

        mu = self.mean_bn(self.mu(h, adj))
        log_var = self.logvar_bn(self.log_var(h, adj))
        return mu, log_var


# class AE_encoder(nn.Module):
#     def __init__(self, omics_dim: int, num_topics: int, hidden_dim: int):
#         super(AE_encoder, self).__init__()
#
#         self.f1 = nn.Linear(omics_dim, hidden_dim)
#         self.f2 = nn.Linear(hidden_dim, hidden_dim)
#         # self.dropout = nn.Dropout(p=0)
#
#         self.mu = nn.Linear(hidden_dim, num_topics)
#         self.log_var = nn.Linear(hidden_dim, num_topics)
#
#         self.mean_bn = nn.BatchNorm1d(num_topics)
#         self.mean_bn.weight.requires_grad = False
#         self.logvar_bn = nn.BatchNorm1d(num_topics)
#         self.logvar_bn.weight.requires_grad = False
#
#     def forward(self, x, adj):
#         h = F.relu(self.f1(x))
#         h = F.relu(self.f2(h))
#
#         mu = self.mean_bn(self.mu(h))
#         log_var = self.logvar_bn(self.log_var(h))
#
#         return mu, log_var
class AE_encoder(nn.Module):
    def __init__(self, omics_dim: int, num_topics: int, hidden_dim: int):
        super(AE_encoder, self).__init__()

        self.f1 = nn.Linear(omics_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)

        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.zeros_(self.f1.bias)
        nn.init.xavier_normal_(self.f2.weight)
        nn.init.zeros_(self.f2.bias)

        self.mu = nn.Linear(hidden_dim, num_topics)
        self.log_var = nn.Linear(hidden_dim, num_topics)

        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.xavier_uniform_(self.log_var.weight)
        nn.init.zeros_(self.log_var.bias)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False

    def forward(self, x, adj):
        h = F.relu(self.f1(x))
        h = F.relu(self.f2(h))

        mu = self.mean_bn(self.mu(h))
        log_var = self.logvar_bn(self.log_var(h))

        return mu, log_var