"""Stock rankers; the input interface has no access to future labels."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width*2, width), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.body(x)


class FactorContext(nn.Module):
    """Permutation-equivariant information aggregation via learned latent factors."""
    def __init__(self, width, factors, dropout):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(factors, width) / width**0.5)
        self.gather = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.distribute = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        daily = self.norm(x)[None]
        factors, _ = self.gather(self.queries[None], daily, daily, need_weights=False)
        update, _ = self.distribute(daily, factors, factors, need_weights=False)
        return x + update[0]


class StockRanker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width, dropout = config.get("width", 128), config.get("dropout", 0.1)
        self.context_enabled = config.get("context", True)
        self.market_gate = config.get("market_gate", False)
        self.temporal = config["family"] == "temporal_mixer"
        self.cs_norm = config.get("cs_norm", False)
        if self.market_gate:
            self.gate = nn.Sequential(nn.Linear(76, 64), nn.SiLU(), nn.Linear(64, 158))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        input_width = 158 + (76 if self.context_enabled and not self.temporal else 0)
        self.project = nn.Linear(input_width, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(config.get("depth", 2))])
        if self.temporal:
            self.time_mix = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
            self.combine = nn.Linear(width*3 + (76 if self.context_enabled else 0), width)
        factors = config.get("latent_factors", 0)
        self.relations = FactorContext(width, factors, dropout) if factors else nn.Identity()
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, stock, context):
        if self.cs_norm:
            stock = (stock-stock.mean(0, keepdim=True)) / stock.std(0, keepdim=True, correction=0).clamp_min(0.1)
        if self.market_gate:
            stock = stock * (1 + torch.tanh(self.gate(context)))[:, None]
        if self.temporal:
            x = self.project(stock)
            x = x + self.time_mix(x.transpose(1, 2)).transpose(1, 2)
            x = self.blocks(x)
            scales = [x[:, -1], x[:, -4:].mean(1), x.mean(1)]
            if self.context_enabled:
                scales.append(context)
            x = self.combine(torch.cat(scales, -1))
        else:
            x = stock[:, -1]
            if self.context_enabled:
                x = torch.cat([x, context], -1)
            x = self.blocks(F.gelu(self.project(x)))
        return self.output(self.relations(x)).squeeze(-1)


class EnsembleLinear(nn.Module):
    def __init__(self, input_width, output_width, members):
        super().__init__()
        self.shared = nn.Linear(input_width, output_width, bias=False)
        self.input_scale = nn.Parameter(torch.empty(members, input_width).normal_(1, 0.1))
        self.output_scale = nn.Parameter(torch.empty(members, output_width).normal_(1, 0.1))
        self.bias = nn.Parameter(torch.zeros(members, output_width))

    def forward(self, x):
        return self.shared(x*self.input_scale) * self.output_scale + self.bias


class BatchEnsembleRanker(nn.Module):
    """BatchEnsemble adaptation inspired by TabM, not an exact paper reproduction."""
    def __init__(self, config):
        super().__init__()
        self.context_enabled = config.get("context", True)
        self.cs_norm = config.get("cs_norm", False)
        members, width = config.get("members", 8), config.get("width", 128)
        dimensions = [234 if self.context_enabled else 158] + [width]*config.get("depth", 3) + [1]
        self.layers = nn.ModuleList([EnsembleLinear(a, b, members) for a, b in zip(dimensions[:-1], dimensions[1:])])
        self.dropout = nn.Dropout(config.get("dropout", 0.1))

    def forward(self, stock, context):
        x = stock[:, -1]
        if self.cs_norm:
            x = (x-x.mean(0)) / x.std(0, correction=0).clamp_min(0.1)
        if self.context_enabled:
            x = torch.cat([x, context], -1)
        x = x[:, None, :]
        for layer in self.layers[:-1]:
            x = self.dropout(F.gelu(layer(x)))
        return self.layers[-1](x).squeeze(-1)


class MasterControl(nn.Module):
    def __init__(self, config):
        super().__init__()
        from models.master import MASTER
        self.model = MASTER(d_feat=158, d_model=config.get("width", 128),
                            t_nhead=4, s_nhead=2, T_dropout_rate=config.get("dropout", 0.3),
                            S_dropout_rate=config.get("dropout", 0.3), beta=5.0)

    def forward(self, stock, context):
        # JKP occupies 13 columns before the 63 market columns. MASTER uses only market gating.
        market = context[:, None, 13:].expand(-1, stock.shape[1], -1)
        return self.model(torch.cat([stock, market], -1))


def make_model(config):
    if config["family"] == "batch_ensemble":
        return BatchEnsembleRanker(config)
    if config["family"] == "master_control":
        return MasterControl(config)
    if config["family"] in ("residual", "temporal_mixer"):
        return StockRanker(config)
    raise ValueError(config["family"])


def rank_loss(predictions, labels, objective):
    scores = predictions.mean(1) if predictions.ndim == 2 else predictions
    targets = labels[:, None].expand_as(predictions) if predictions.ndim == 2 else labels
    mse = F.mse_loss(predictions, targets)
    if objective == "mse":
        return mse
    centered_scores = scores-scores.mean()
    centered_labels = labels-labels.mean()
    correlation = (centered_scores*centered_labels).sum() / (
        centered_scores.square().sum().sqrt()*centered_labels.square().sum().sqrt()).clamp_min(1e-6)
    if objective == "corr":
        return 1-correlation + 0.05*mse
    if objective == "mixed":
        return 0.5*mse + 0.5*(1-correlation)
    if objective == "tail_pair":
        first = torch.randint(len(scores), (min(4096, len(scores)*8),), device=scores.device)
        second = torch.randint(len(scores), first.shape, device=scores.device)
        direction = (labels[first]-labels[second]).sign()
        weights = 1 + 2*(torch.maximum(labels[first], labels[second]) > 1.0).float()
        pair = F.softplus(-direction*(scores[first]-scores[second]))
        return 0.3*mse + 0.3*(1-correlation) + 0.4*(pair*weights).mean()
    if objective == "listwise":
        distribution = F.softmax(labels / 0.7, dim=0)
        return 0.3*mse + 0.7*(-distribution*F.log_softmax(scores/0.7, dim=0)).sum()
    raise ValueError(objective)
