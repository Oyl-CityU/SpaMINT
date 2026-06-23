import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from torch import Tensor
from .components import GHN_encoder, decoder, Decoder_graph, GCN_encoder, AE_encoder
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from collections import defaultdict


class SpaMINT(nn.Module):
    def __init__(
            self,
            omics_data: Dict[str, Any],
            device=torch.device('cpu'),
            num_topics: int = 20,
            embedding_dim: int = 400,
            hidden_dim: int = 200,
            lambda_graph: float = 1.0,
            weight_loss_ECR: Optional[List[float]] = None,
            sinkhorn_alpha: int = 20,
            beta_temp: float = 0.05,
            OT_max_iter: int = 1000,
            random_seed: int = 2025,
            weight_decay: float = 0.00,
            learning_rate: float = 8e-3,
            epochs: int = 800,
            network: str = 'GHN',
            verbose: bool = False
    ) -> None:

        super(SpaMINT, self).__init__()

        self.device = device

        # Validate input data
        self.modalities = list(omics_data.keys())

        n_spots = {key: adata.n_obs for key, adata in omics_data.items()}
        if len(set(n_spots.values())) > 1:
            raise ValueError(f"Error: Sample counts are inconsistent across omics! {n_spots}")
        self.n_spots = next(iter(n_spots.values()))

        # Initialize dimensions and parameters
        self.feat_dims = {key: adata.obsm['feat'].shape[1] for key, adata in omics_data.items()}
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lambda_graph = lambda_graph
        self.weight_loss_ECR = weight_loss_ECR or [1.0] * len(self.modalities)
        self.sinkhorn_alpha = sinkhorn_alpha
        self.beta_temp = beta_temp
        self.OT_max_iter = OT_max_iter
        self.random_seed = random_seed
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.network = network

        self.reconstruction_omics: Dict[str, Tensor] = {
            modality: torch.zeros(self.n_spots, dim) for modality, dim in self.feat_dims.items()
        }
        self._register_omics_data(omics_data)
        self._init_topic_embeddings()
        self._build_enc_dec()
        self._init_prior()
        self._init_graph_decoder()

        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lr_scheduler = self._make_lr_scheduler()

    def _register_omics_data(self, omics_data: Dict[str, Any]) -> None:
        """Register omics data as buffer tensors."""
        self.features = []
        self.adjs = []

        for modality in self.modalities:
            adata = omics_data[modality]

            # Register features
            feat = torch.tensor(adata.obsm['feat'].copy(), dtype=torch.float32, device=self.device)
            self.register_buffer(f"{modality}_feat", feat)
            self.features.append(getattr(self, f"{modality}_feat"))

            # Register spatial adjacency matrices
            adj = adata.obsm['adj'].to(self.device)
            self.register_buffer(f"{modality}_adj", adj)
            self.adjs.append(getattr(self, f"{modality}_adj"))

        adj_label = omics_data[self.modalities[0]].obsm['adj_label'].to(self.device)
        self.register_buffer("adj_label", adj_label)

    def _init_topic_embeddings(self) -> None:
        """Initialize topic embeddings with truncated normal distribution."""
        self.topic_embeddings = torch.empty((self.num_topics, self.embedding_dim))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings), requires_grad=True)

    def _build_enc_dec(self):
        """Build encoder and decoder modules."""

        # Encoders
        if self.network == "GHN":
            self.encoders = nn.ModuleList([
                GHN_encoder(self.n_spots, self.num_topics, self.hidden_dim)
                for _ in self.feat_dims.values()
            ])
        elif self.network == "GCN":
            self.encoders = nn.ModuleList([
                GCN_encoder(dim, self.num_topics, self.hidden_dim)
                for dim in self.feat_dims.values()
            ])
        elif self.network == "AE":
            self.encoders = nn.ModuleList([
                AE_encoder(dim, self.num_topics, self.hidden_dim)
                for dim in self.feat_dims.values()
            ])
        else:
            print("Network {} is not supported".format(self.network))

        # Decoders
        self.decoders = nn.ModuleList([
            decoder(
                omics_dim=dim,
                num_topics=self.num_topics,
                topic_embeddings=self.topic_embeddings,
                weight_loss_ECR=self.weight_loss_ECR[i],
                sinkhorn_alpha=self.sinkhorn_alpha,
                OT_max_iter=self.OT_max_iter,
                beta_temp=self.beta_temp
            )
            for i, dim in enumerate(self.feat_dims.values())
        ])

    def _init_prior(self) -> None:
        """Initialize prior distribution parameters."""

        # Prior parameters (Dirichlet approximation)
        a = 1 * np.ones((1, self.num_topics)).astype(np.float32)
        self.mu_prior = nn.Parameter(
            (torch.as_tensor((np.log(a).T - np.mean(np.log(a), 1)).T))
            .unsqueeze(0)
            .expand(1, self.n_spots, self.num_topics),
            requires_grad=False
        ).to(self.device)
        self.var_prior = nn.Parameter(
            (torch.as_tensor((((1.0 / a) * (1 - (2.0 / self.num_topics))).T +
                              (1.0 / (self.num_topics * self.num_topics)) * np.sum(1.0 / a, 1)).T))
            .unsqueeze(0)
            .expand(1, self.n_spots, self.num_topics),
            requires_grad=False
        ).to(self.device)

    def _init_graph_decoder(self) -> None:
        """Initialize graph decoder components."""
        self.decoder_graph = Decoder_graph()
        self.alpha = nn.Parameter(torch.tensor(0.2, dtype=torch.float32, device=self.device))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for Gaussian sampling."""
        if self.training:
            # torch.use_deterministic_algorithms(True)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encoder(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode spatial multi-omics data into topic distributions."""

        mu_list = []
        logvar_list = []

        # Encode each omics modality
        for encoder, feat, adj in zip(self.encoders, self.features, self.adjs):
            mu, logvar = encoder(feat, adj)
            mu_list.append(mu.unsqueeze(0))
            logvar_list.append(logvar.unsqueeze(0))

        if len(self.modalities) == 1:
            mu = mu_list[0].squeeze(0)
            logvar = logvar_list[0].squeeze(0)
        else:

            # Concatenate the prior with each modality's posterior parameters.
            Mu = torch.cat([self.mu_prior] + mu_list, dim=0)
            Log_var = torch.cat([self.var_prior.log()] + logvar_list, dim=0)

            # Fuse multi-omics information
            weights = torch.ones(Mu.size(0), device=self.device)
            mu, logvar = self._expert_fusion(Mu, Log_var, weights)

        # Generate topic distribution
        theta_raw = self.reparameterize(mu, logvar)
        theta = F.softmax(theta_raw, dim=-1)
        loss_KL = self._kl_divergence(mu, logvar)

        return theta, loss_KL, theta_raw

    @staticmethod
    def _expert_fusion(
            mu: torch.Tensor,
            logvar: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse multiple Gaussian experts using product-of-experts."""
        var = torch.exp(logvar) + eps
        num_experts = mu.shape[0]

        # Handle weights
        weights = weights if weights is not None else torch.ones(num_experts, device=mu.device, dtype=mu.dtype)
        assert weights.shape == (num_experts,), f"Invalid weight shape: {weights.shape} vs ({num_experts},)"

        # Compute precision-weighted fusion
        weighted_precision = weights.view(-1, 1, 1) / (var + eps)
        sum_weighted_precision = torch.sum(weighted_precision, dim=0)

        pd_mu = torch.sum(mu * weighted_precision, dim=0) / (sum_weighted_precision + eps)
        pd_var = 1.0 / (sum_weighted_precision + eps)
        pd_logvar = torch.log(pd_var + eps)

        return pd_mu, pd_logvar

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""

        logvar = logvar.unsqueeze(0)
        mu = mu.unsqueeze(0)
        var = torch.exp(logvar)
        var_division = var / self.var_prior
        diff = mu - self.mu_prior
        diff_term = diff * diff / self.var_prior
        logvar_division = self.var_prior.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(-1) - self.num_topics)
        return KLD.mean()

    def _calc_weight(self,
                     epoch: int,
                     cutoff_ratio: float = 0.,
                     warmup_ratio: float = 1 / 5,  # 1 / 5
                     min_weight: float = 0.,
                     max_weight: float = 2e-2
                     ) -> float:
        """Calculate KL weight with warmup schedule."""
        fully_warmup_epoch = self.epochs * warmup_ratio

        if epoch < self.epochs * cutoff_ratio:
            return 0.
        if warmup_ratio:
            return max(min(1., epoch / fully_warmup_epoch) * max_weight, min_weight)
        else:
            return max_weight

    def get_theta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get topic distribution \theta"""

        theta, loss_KL, theta_raw = self.encoder()
        return (theta, loss_KL) if self.training else (theta, theta_raw)

    def forward(self, epoch: int = 0) -> Dict[str, torch.Tensor]:
        """Full forward pass with loss calculation."""
        theta, loss_KL, Z = self.encoder()

        recon_loss_total = 0.0
        loss_ECR_total = 0.0
        rst_dict: Dict[str, torch.Tensor] = {}

        # Compute loss for each modality.
        for local_decoder, feat, modality in zip(self.decoders, self.features, self.modalities):
            rst = local_decoder(theta=theta, omics=feat)
            # recon_loss_total += rst['recon_loss']
            # loss_ECR_total += rst['loss_ECR']
            # rst_dict[f'recon_loss_{modality}'] = rst['recon_loss']

            recon_loss = rst['recon_loss']
            if modality == 'ADT' and 100 > self.feat_dims['ADT'] > 0:
                recon_loss *= 2
            elif modality == 'ADT' and 1000 > self.feat_dims['ADT'] >= 100:
                recon_loss *= 1.5

            recon_loss_total += recon_loss
            loss_ECR_total += rst['loss_ECR']
            rst_dict[f'recon_loss_{modality}'] = recon_loss
            self.reconstruction_omics[modality] = rst['recon_omics']

        # KL weighting with warmup
        KL_weight = self._calc_weight(epoch)
        loss_TM = recon_loss_total + KL_weight * loss_KL

        # Graph reconstruction loss
        adj_label = getattr(self.adj_label, "to_dense", lambda: self.adj_label)().view(-1)
        loss_Graph = F.binary_cross_entropy(self.decoder_graph(Z, self.alpha).view(-1), adj_label)

        # Total loss
        loss = loss_TM + loss_ECR_total + self.lambda_graph * loss_Graph

        rst_dict.update({
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR_total': loss_ECR_total,
            'loss_Graph': loss_Graph
        })

        return rst_dict

    def _make_lr_scheduler(self, ):
        lr_scheduler = StepLR(self.optimizer, step_size=100, gamma=0.8)  # 0.5
        return lr_scheduler

    def train_model(self) -> Dict[str, np.ndarray]:
        self.to(self.device)

        # Initialize progress bar for non-verbose mode
        pbar = tqdm(total=self.epochs, disable=self.verbose)

        for epoch in range(1, self.epochs + 1):
            # Single epoch training
            loss_dict = self._train_step(epoch)

            # Update display based on verbosity
            if self.verbose:
                self._print_verbose_loss(epoch, loss_dict)
            else:
                pbar.set_postfix({'loss': loss_dict['loss']})
                pbar.update()

        pbar.close()
        return self._collect_outputs()

    def _train_step(self, epoch: int) -> Dict[str, float]:
        """Complete one training epoch"""

        self.train()
        loss_rst_dict = defaultdict(float)

        loss_dict = self.forward(epoch)
        total_loss = loss_dict['loss']

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for key in loss_dict:
            loss_rst_dict[key] += loss_dict[key]

        self.lr_scheduler.step()

        return {k: v for k, v in loss_rst_dict.items()}

    @staticmethod
    def _print_verbose_loss(epoch: int, loss_dict: Dict[str, float]):
        """Detailed loss printing for verbose mode"""

        loss_str = f"Epoch {epoch:03d}"
        for k, v in loss_dict.items():
            loss_str += f" | {k}: {v:.3f}"

        print(loss_str)

    def _collect_outputs(self) -> Dict[str, np.ndarray]:
        self.eval()
        outputs = {
            'Topic_proportion': self.get_theta()[0].detach().cpu().numpy(),
            'theta_raw': self.get_theta()[1].detach().cpu().numpy(),
            'Topic_embeddings': self.topic_embeddings.detach().cpu().numpy(),
        }

        for modality, local_decoder in zip(self.modalities, self.decoders):
            outputs.update({
                f"{modality}_embeddings": local_decoder.omics_embeddings.detach().cpu().numpy(),
                f"Beta_{modality}": local_decoder.get_beta().detach().cpu().numpy(),
            })

        for modality, recon_omic in self.reconstruction_omics.items():
            outputs.update({
                f"recon_{modality}": recon_omic.detach().cpu().numpy(),
            })

        return outputs
