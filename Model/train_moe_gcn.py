#!/usr/bin/env python3
"""
Training script for a multi-atlas GCN Mixture-of-Experts model.

Each atlas provides a structural connectivity (SC) matrix, treated as the graph
adjacency, and a functional connectivity (FC) matrix, treated as node features.
Atlas-specific encoders generate embeddings that are fed into shared and
atlas-specific experts. A gating network combines expert logits, enabling the
model to learn atlas preferences for CN vs MCI classification.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from adni_dataloader import ADNIDataset


DEFAULT_ATLAS_ENCODER_CONFIG: Dict[str, Dict[str, float]] = {
    "AAL": {
        "hidden_dim": 32,
        "embedding_dim": 32,
        "dropout": 0.2,
        "attention_heads": 4,
    },
    "3Hinge": {
        "hidden_dim": 128,
        "embedding_dim": 128,
        "dropout": 0.2,
        "attention_heads": 4,
    },
    "Destrieux": {
        "hidden_dim": 32,
        "embedding_dim": 32,
        "dropout": 0.2,
        "attention_heads": 4,
    },
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetrize structural connectivity matrices and ensure self-loops.
    """
    adj = torch.relu(adj)
    adj = 0.5 * (adj + adj.transpose(1, 2))
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
    adj = adj + eye
    return adj


class GraphAttentionLayer(nn.Module):
    """
    Dense graph attention layer inspired by GAT (Veličković et al., 2018).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        self.linear = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, out_features))
        self.bias = nn.Parameter(
            torch.zeros(num_heads * out_features if concat else out_features)
        )

        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn_src, gain=1.414)
        nn.init.xavier_uniform_(self.attn_dst, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, num_nodes, in_features]
            adj: Structural connectivity matrix with self-loops [batch, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.shape

        h = self.linear(x)
        h = h.view(batch_size, num_nodes, self.num_heads, self.out_features)

        attn_src = (h * self.attn_src.view(1, 1, self.num_heads, self.out_features)).sum(-1)
        attn_dst = (h * self.attn_dst.view(1, 1, self.num_heads, self.out_features)).sum(-1)

        logits = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)
        logits = self.leaky_relu(logits)

        # Encourage strong SC edges via log(1 + weight)
        logits = logits + torch.log1p(adj).unsqueeze(-1)

        mask = adj > 0
        attention = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attention = torch.softmax(attention, dim=2)
        attention = self.dropout(attention)
        self.last_attention = attention.detach().cpu()

        h_prime = torch.einsum("bijh,bjhf->bihf", attention, h)

        if self.concat:
            h_prime = h_prime.reshape(batch_size, num_nodes, self.num_heads * self.out_features)
        else:
            h_prime = h_prime.mean(dim=2)

        h_prime = h_prime + self.bias
        return h_prime


class AtlasGraphEncoder(nn.Module):
    """Encodes an atlas graph into a fixed-size embedding using graph attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float = 0.3,
        attention_heads: int = 4,
        second_layer_heads: int = 1,
        head_fusion: str = "mean",
    ):
        super().__init__()
        if head_fusion not in {"mean", "concat"}:
            raise ValueError("head_fusion must be 'mean' or 'concat'")

        fuse_by_concat = head_fusion == "concat"

        self.attn1 = GraphAttentionLayer(
            in_features=input_dim,
            out_features=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            concat=fuse_by_concat,
        )
        attn2_input_dim = hidden_dim * attention_heads if fuse_by_concat else hidden_dim
        self.attn2 = GraphAttentionLayer(
            in_features=attn2_input_dim,
            out_features=embedding_dim,
            num_heads=second_layer_heads,
            dropout=dropout,
            concat=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.attn1(node_features, adj)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.attn2(h, adj)
        h = F.elu(h)
        embedding = h.mean(dim=1)
        return embedding


class ExpertHead(nn.Module):
    """Two-layer MLP expert that produces a single logit."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiAtlasMoE(nn.Module):
    """
    Mixture-of-Experts model with shared and atlas-specific experts.
    """

    def __init__(
        self,
        atlas_shapes: Dict[str, Dict[str, Tuple[int, ...]]],
        embedding_dim: int = 64,
        encoder_hidden_dim: int = 128,
        expert_hidden_dim: int = 128,
        shared_experts: int = 2,
        dropout: float = 0.3,
        attention_heads: int = 4,
        second_layer_heads: int = 1,
        head_fusion: str = "mean",
        atlas_encoder_configs: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        super().__init__()
        self.atlases = sorted(atlas_shapes.keys())

        self.atlas_encoder_configs = atlas_encoder_configs or {}
        self.atlas_embedding_dims: Dict[str, int] = {}
        self.atlas_encoder_dropout: Dict[str, float] = {}

        self.encoders = nn.ModuleDict()
        for atlas in self.atlases:
            fc_shape = atlas_shapes[atlas]["FC"]
            node_dim = fc_shape[1] if len(fc_shape) > 1 else fc_shape[0]
            config = self.atlas_encoder_configs.get(atlas, {})
            atlas_hidden_dim = int(config.get("hidden_dim", encoder_hidden_dim))
            atlas_embedding_dim = int(config.get("embedding_dim", embedding_dim))
            atlas_dropout = float(config.get("dropout", dropout))
            atlas_attention_heads = int(config.get("attention_heads", attention_heads))
            atlas_second_layer_heads = int(config.get("second_layer_heads", second_layer_heads))

            self.encoders[atlas] = AtlasGraphEncoder(
                input_dim=node_dim,
                hidden_dim=atlas_hidden_dim,
                embedding_dim=atlas_embedding_dim,
                dropout=atlas_dropout,
                attention_heads=atlas_attention_heads,
                second_layer_heads=atlas_second_layer_heads,
                head_fusion=head_fusion,
            )

            self.atlas_embedding_dims[atlas] = atlas_embedding_dim
            self.atlas_encoder_dropout[atlas] = atlas_dropout

        total_embedding_dim = sum(self.atlas_embedding_dims.values())

        self.shared_experts = nn.ModuleList(
            [
                ExpertHead(total_embedding_dim, expert_hidden_dim, dropout=dropout)
                for _ in range(shared_experts)
            ]
        )

        self.specific_experts = nn.ModuleDict(
            {
                atlas: ExpertHead(
                    self.atlas_embedding_dims[atlas],
                    expert_hidden_dim,
                    dropout=self.atlas_encoder_dropout[atlas],
                )
                for atlas in self.atlases
            }
        )

        self.alignment_heads = nn.ModuleDict(
            {
                atlas: nn.Linear(self.atlas_embedding_dims[atlas], 1)
                for atlas in self.atlases
            }
        )

        self.expert_names = [f"shared_{i}" for i in range(len(self.shared_experts))]
        self.expert_names += [f"specific_{atlas}" for atlas in self.atlases]

        total_experts = len(self.shared_experts) + len(self.atlases)
        self.gate = nn.Sequential(
            nn.Linear(total_embedding_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, total_experts),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        embeddings = []
        per_atlas_embeddings = {}

        for atlas in self.atlases:
            fc = batch[f"{atlas}_FC"].float()
            sc = batch[f"{atlas}_SC"].float()
            adj = prepare_adjacency(sc)
            embedding = self.encoders[atlas](fc, adj)
            embeddings.append(embedding)
            per_atlas_embeddings[atlas] = embedding

        shared_input = torch.cat(embeddings, dim=-1)

        shared_logits = [expert(shared_input) for expert in self.shared_experts]
        specific_logits = [
            self.specific_experts[atlas](per_atlas_embeddings[atlas]) for atlas in self.atlases
        ]

        expert_logits = torch.cat(shared_logits + specific_logits, dim=-1)
        gate_logits = self.gate(shared_input)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused_logit = torch.sum(expert_logits * gate_weights, dim=-1)

        alignment_logits = {
            atlas: self.alignment_heads[atlas](per_atlas_embeddings[atlas])
            for atlas in self.atlases
        }

        aux = {
            "expert_logits": expert_logits,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "atlas_embeddings": per_atlas_embeddings,
            "alignment_logits": alignment_logits,
        }
        return fused_logit, aux


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits).cpu()
    preds = (probs >= 0.5).long()
    labels_long = labels.long().cpu()

    correct = (preds == labels_long).sum().item()
    total = labels_long.numel()
    accuracy = correct / total if total > 0 else 0.0

    tp = ((preds == 1) & (labels_long == 1)).sum().item()
    tn = ((preds == 0) & (labels_long == 0)).sum().item()
    fp = ((preds == 1) & (labels_long == 0)).sum().item()
    fn = ((preds == 0) & (labels_long == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    try:
        metrics["roc_auc"] = roc_auc_score(labels_long.numpy(), probs.numpy())
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = average_precision_score(labels_long.numpy(), probs.numpy())
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    collect_gating: bool = False,
    alignment_weight: float = 0.0,
    balance_weight: float = 0.0,
):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    gating_weights = []
    gating_labels = []
    alignment_loss_total = 0.0
    balance_loss_total = 0.0
    per_atlas_alignment_sums: Optional[Dict[str, float]] = None

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"].float()
        inputs = {k: v for k, v in batch.items() if k not in {"label", "subject_id", "dx", "mmse"}}
        batch_size = labels.size(0)

        logits, aux = model(inputs)
        loss = criterion(logits, labels)
        alignment_loss = None
        balance_loss = None

        alignment_losses = []
        if aux["alignment_logits"]:
            if per_atlas_alignment_sums is None:
                per_atlas_alignment_sums = {atlas: 0.0 for atlas in aux["alignment_logits"].keys()}
            for atlas, logit in aux["alignment_logits"].items():
                atlas_loss = F.binary_cross_entropy_with_logits(logit.view(-1), labels)
                alignment_losses.append(atlas_loss)
                per_atlas_alignment_sums[atlas] += atlas_loss.detach().item() * batch_size
        if alignment_weight > 0.0 and alignment_losses:
            alignment_loss = torch.stack(alignment_losses).mean()
            loss = loss + alignment_weight * alignment_loss
            # loss = loss
            alignment_loss_total += alignment_loss.detach().item() * batch_size

        if balance_weight > 0.0:
            gate_w = aux["gate_weights"]
            expert_target = torch.full(
                (gate_w.size(-1),),
                1.0 / gate_w.size(-1),
                device=gate_w.device,
                dtype=gate_w.dtype,
            )
            balance_loss = F.mse_loss(gate_w.mean(dim=0), expert_target)
            loss = loss + balance_weight * balance_loss
            # loss = loss

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        if balance_loss is not None:
            balance_loss_total += balance_loss.item() * batch_size

        if collect_gating:
            gating_weights.append(aux["gate_weights"].detach().cpu())
            gating_labels.append(labels.detach().cpu())

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    metrics = compute_classification_metrics(logits_tensor, labels_tensor)
    metrics["loss"] = total_loss / max(total_samples, 1)
    if alignment_weight > 0.0 and total_samples > 0:
        metrics["alignment_loss"] = alignment_loss_total / total_samples
    if balance_weight > 0.0 and total_samples > 0:
        metrics["balance_loss"] = balance_loss_total / total_samples
    if per_atlas_alignment_sums and total_samples > 0:
        for atlas, loss_sum in per_atlas_alignment_sums.items():
            metrics[f"alignment_loss_{atlas}"] = loss_sum / total_samples

    gating_summary = None
    if collect_gating and gating_weights:
        gating_tensor = torch.cat(gating_weights)
        label_tensor = torch.cat(gating_labels).long()
        gating_summary = defaultdict(dict)
        for idx, name in enumerate(model.expert_names):
            for cls in (0, 1):
                mask = label_tensor == cls
                if mask.any():
                    gating_summary[name][cls] = gating_tensor[mask, idx].mean().item()

    return metrics, gating_summary


def format_metrics(split: str, epoch: int, metrics: Dict[str, float]) -> str:
    parts = [
        f"{split} Epoch {epoch:03d}",
        f"Loss {metrics['loss']:.4f}",
        f"Acc {metrics['accuracy']:.4f}",
        f"Prec {metrics['precision']:.4f}",
        f"Recall {metrics['recall']:.4f}",
        f"F1 {metrics['f1']:.4f}",
    ]
    if "roc_auc" in metrics:
        parts.append(f"ROC {metrics['roc_auc']:.4f}")
    if "pr_auc" in metrics:
        parts.append(f"PR {metrics['pr_auc']:.4f}")
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-atlas GCN MoE classifier.")
    parser.add_argument("--train-json", default="/mnt/disk3/Multi_Atlas/ADNI_train_no_mmse_clean_complete.json", help="Path to training JSON file.")
    parser.add_argument("--test-json", default="/mnt/disk3/Multi_Atlas/ADNI_test_no_mmse_clean_complete.json", help="Path to test/validation JSON file.")
    parser.add_argument("--atlases", default=None, help="Comma-separated list of atlas names to use.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--expert-hidden-dim", type=int, default=128)
    parser.add_argument("--shared-experts", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--second-layer-heads", type=int, default=1)
    parser.add_argument(
        "--head-fusion",
        choices=["mean", "concat"],
        default="mean",
        help="How to fuse multi-head outputs from the first attention layer.",
    )
    parser.add_argument(
        "--use-default-atlas-config",
        action="store_true",
        help="Use built-in atlas-specific GAT hyperparameters derived from grid search.",
    )
    parser.add_argument(
        "--atlas-config",
        default=None,
        help="Optional JSON file with atlas-specific encoder settings (overrides defaults).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of cross-validation folds (0 or 1 disables CV and uses train/test).",
    )
    parser.add_argument(
        "--cv-seed",
        type=int,
        default=42,
        help="Random seed for cross-validation shuffling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model initialization, dataloader shuffling, etc.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to save training output as text.",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Path to save the best model weights (train/test mode).",
    )
    parser.add_argument(
        "--save-model-dir",
        default=None,
        help="Directory to save best weights per fold during cross-validation.",
    )
    parser.add_argument(
        "--save-training-artefacts",
        default=None,
        help="Directory to dump logs, metrics, and intermediate checkpoints for the current run.",
    )
    parser.add_argument(
        "--alignment-weight",
        type=float,
        default=0.5,
        help="Weight for atlas-wise label alignment auxiliary loss.",
    )
    parser.add_argument(
        "--balance-weight",
        type=float,
        default=0.5,
        help="Weight for expert balance regularizer to discourage gate collapse.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Manual positive-class weight for BCE loss. Defaults to neg/pos from training data.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def create_dataloaders(
    train_json: str,
    test_json: str,
    atlases: Optional[Iterable[str]],
    batch_size: int,
    num_workers: int,
) -> Tuple[ADNIDataset, DataLoader, ADNIDataset, DataLoader]:
    train_dataset = ADNIDataset(train_json, atlases=atlases)
    test_dataset = ADNIDataset(test_json, atlases=atlases)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_dataset, train_loader, test_dataset, test_loader


def compute_class_counts(dataset: ADNIDataset) -> Tuple[int, int]:
    """Return (negatives, positives) class counts for CN/MCI."""
    label_ids = [item["DX"] for item in dataset.data]
    transformed = dataset.label_encoder.transform(label_ids)
    counts = Counter(transformed)
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)
    return neg_count, pos_count


def compute_class_counts_from_indices(dataset: ADNIDataset, indices: Iterable[int]) -> Tuple[int, int]:
    labels = [dataset.data[i]["DX"] for i in indices]
    transformed = dataset.label_encoder.transform(labels)
    counts = Counter(transformed)
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)
    return neg_count, pos_count


def build_atlas_encoder_configs(
    selected_atlases: Iterable[str],
    use_default: bool,
    config_path: Optional[str],
) -> Dict[str, Dict[str, float]]:
    configs: Dict[str, Dict[str, float]] = {}
    if use_default:
        configs.update(DEFAULT_ATLAS_ENCODER_CONFIG)
    if config_path:
        with open(config_path, "r") as f:
            user_configs = json.load(f)
        configs.update({str(k): v for k, v in user_configs.items()})
    return {
        atlas: configs[atlas]
        for atlas in selected_atlases
        if atlas in configs
    }


def train_single_split(
    args,
    log,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    neg_count: int,
    pos_count: int,
    atlas_shapes: Dict[str, Dict[str, Tuple[int, ...]]],
    atlas_encoder_configs: Dict[str, Dict[str, float]],
    prefix: str = "",
    save_path: Optional[str] = None,
    artefact_dir: Optional[str] = None,
) -> Dict[str, float]:
    if args.pos_weight is not None:
        pos_weight_value = args.pos_weight
    else:
        pos_weight_value = neg_count / max(pos_count, 1) if pos_count > 0 else 1.0

    pos_weight_tensor = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)

    model = MultiAtlasMoE(
        atlas_shapes=atlas_shapes,
        embedding_dim=args.embedding_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        shared_experts=args.shared_experts,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
        second_layer_heads=args.second_layer_heads,
        head_fusion=args.head_fusion,
        atlas_encoder_configs=atlas_encoder_configs,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    log(
        f"{prefix}Class counts -> CN: {neg_count}, MCI: {pos_count}; "
        f"using pos_weight={pos_weight_value:.4f}"
    )

    best_val_acc = float("-inf")
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, float]] = None

    def print_atlas_losses(split: str, metrics: Dict[str, float]) -> None:
        atlas_keys = sorted(
            k for k in metrics.keys()
            if k.startswith("alignment_loss_") and k != "alignment_loss"
        )
        if atlas_keys:
            formatted = ", ".join(
                f"{k.replace('alignment_loss_', '')}: {metrics[k]:.4f}"
                for k in atlas_keys
            )
            log(f"{prefix}   Atlas BCE ({split}): {formatted}")

    best_state = None
    if artefact_dir:
        os.makedirs(artefact_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            collect_gating=False,
            alignment_weight=args.alignment_weight,
            balance_weight=args.balance_weight,
        )
        val_metrics, gating_summary = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            collect_gating=True,
            alignment_weight=args.alignment_weight,
            balance_weight=args.balance_weight,
        )

        if artefact_dir:
            epoch_path = os.path.join(artefact_dir, f"epoch_{epoch:03d}.pt")
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, epoch_path)

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            best_val_metrics = val_metrics.copy()
            if save_path:
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        log(f"{prefix}{format_metrics('Train', epoch, train_metrics)}")
        log(f"{prefix}{format_metrics('Val', epoch, val_metrics)}")
        if "alignment_loss" in train_metrics:
            log(f"{prefix}   Aux alignment loss (train): {train_metrics['alignment_loss']:.4f}")
        if "alignment_loss" in val_metrics:
            log(f"{prefix}   Aux alignment loss (val)  : {val_metrics['alignment_loss']:.4f}")
        if "balance_loss" in train_metrics:
            log(f"{prefix}   Balance loss (train)     : {train_metrics['balance_loss']:.4f}")
        if "balance_loss" in val_metrics:
            log(f"{prefix}   Balance loss (val)       : {val_metrics['balance_loss']:.4f}")

        print_atlas_losses("train", train_metrics)
        print_atlas_losses("val", val_metrics)

    if gating_summary:
        cn_pref = ", ".join(
            f"{name}: {values.get(0, 0.0):.3f}" for name, values in gating_summary.items()
        )
        mci_pref = ", ".join(
            f"{name}: {values.get(1, 0.0):.3f}" for name, values in gating_summary.items()
        )
        log(f"{prefix}   Avg gate weight CN : {cn_pref}")
        log(f"{prefix}   Avg gate weight MCI: {mci_pref}")

    assert best_val_metrics is not None
    log(
        f"{prefix}\nBest validation accuracy {best_val_acc:.4f} at epoch {best_epoch:03d} "
        f"(Precision {best_val_metrics['precision']:.4f}, "
        f"Recall {best_val_metrics['recall']:.4f}, "
        f"F1 {best_val_metrics['f1']:.4f}, "
        f"ROC {best_val_metrics.get('roc_auc', float('nan')):.4f}, "
        f"PR {best_val_metrics.get('pr_auc', float('nan')):.4f})"
    )
    best_val_metrics["best_epoch"] = best_epoch
    best_val_metrics["best_accuracy"] = best_val_acc
    if save_path and best_state is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, save_path)
        log(f"{prefix}Saved best model weights to {save_path}")
    return best_val_metrics


def main():
    args = parse_args()
    atlases = [atlas.strip() for atlas in args.atlases.split(",")] if args.atlases else None

    log_handle = open(args.log_file, "w") if args.log_file else None

    def log(message: str) -> None:
        print(message)
        if log_handle:
            log_handle.write(message + "\n")
            log_handle.flush()

    try:
        device = torch.device(args.device)
        set_global_seed(args.seed)

        if args.cv_folds and args.cv_folds > 1:
            dataset = ADNIDataset(args.train_json, atlases=atlases)
            selected_atlases = getattr(dataset, "selected_atlases", [])
            atlas_encoder_configs = build_atlas_encoder_configs(
                selected_atlases,
                args.use_default_atlas_config,
                args.atlas_config,
            )
            if atlas_encoder_configs:
                formatted = ", ".join(
                    f"{atlas}: {cfg}" for atlas, cfg in atlas_encoder_configs.items()
                )
                log(f"Using atlas-specific encoder configs -> {formatted}")

            kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.cv_seed)
            fold_metrics: List[Dict[str, float]] = []

            if args.save_training_artefacts:
                os.makedirs(args.save_training_artefacts, exist_ok=True)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
                prefix = f"[Fold {fold_idx + 1}/{args.cv_folds}] "
                log("=" * 80)
                log(f"{prefix}Training with {len(train_idx)} samples, validating on {len(val_idx)} samples")

                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)

                train_loader = DataLoader(
                    train_subset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                )

                neg_count, pos_count = compute_class_counts_from_indices(dataset, train_idx)
                save_path = None
                if args.save_model_dir:
                    os.makedirs(args.save_model_dir, exist_ok=True)
                    save_path = os.path.join(
                        args.save_model_dir,
                        f"cv_fold{fold_idx + 1}_best.pt",
                    )

                artefact_dir = (
                    os.path.join(args.save_training_artefacts, f"fold_{fold_idx + 1}")
                    if args.save_training_artefacts
                    else None
                )

                best_metrics = train_single_split(
                    args,
                    log,
                    device,
                    train_loader,
                    val_loader,
                    neg_count,
                    pos_count,
                    dataset.atlas_shapes,
                    atlas_encoder_configs,
                    prefix=prefix,
                    save_path=save_path,
                    artefact_dir=artefact_dir,
                )
                best_metrics["fold"] = fold_idx + 1
                fold_metrics.append(best_metrics)

            if fold_metrics:
                avg_acc = sum(m["best_accuracy"] for m in fold_metrics) / len(fold_metrics)
                avg_f1 = sum(m["f1"] for m in fold_metrics) / len(fold_metrics)
                best_metric = max(fold_metrics, key=lambda m: m["best_accuracy"])
                log(
                    f"\nBest fold -> Fold {best_metric['fold']} accuracy {best_metric['best_accuracy']:.4f}, "
                    f"precision {best_metric['precision']:.4f}, recall {best_metric['recall']:.4f}, F1 {best_metric['f1']:.4f}"
                )
                log(
                    f"\nCross-validation summary -> Avg best accuracy {avg_acc:.4f}, Avg F1 {avg_f1:.4f}"
                )

        else:
            train_dataset, train_loader, test_dataset, test_loader = create_dataloaders(
                args.train_json,
                args.test_json,
                atlases=atlases,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            selected_atlases = getattr(train_dataset, "selected_atlases", [])
            atlas_encoder_configs = build_atlas_encoder_configs(
                selected_atlases,
                args.use_default_atlas_config,
                args.atlas_config,
            )
            if atlas_encoder_configs:
                formatted = ", ".join(
                    f"{atlas}: {cfg}" for atlas, cfg in atlas_encoder_configs.items()
                )
                log(f"Using atlas-specific encoder configs -> {formatted}")

            neg_count, pos_count = compute_class_counts(train_dataset)
            save_path = args.save_model
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if args.save_training_artefacts:
                os.makedirs(args.save_training_artefacts, exist_ok=True)

            train_single_split(
                args,
                log,
                device,
                train_loader,
                test_loader,
                neg_count,
                pos_count,
                train_dataset.atlas_shapes,
                atlas_encoder_configs,
                save_path=save_path,
                artefact_dir=args.save_training_artefacts,
            )

    finally:
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    main()
