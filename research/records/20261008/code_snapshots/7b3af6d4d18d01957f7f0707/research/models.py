"""Stock rankers; the input interface has no access to future labels."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import ARTIFACTS
from .common import config_id
from .covariance import low_rank_gaussian_nll
from .numerical import make_feature_encoder
from .ranking import topk_membership_loss


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
        self.feature_encoder = make_feature_encoder(config)
        if self.market_gate:
            self.gate = nn.Sequential(nn.Linear(76, 64), nn.SiLU(), nn.Linear(64, 158))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        input_width = self.feature_encoder.output_features + (76 if self.context_enabled and not self.temporal else 0)
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
        stock = self.feature_encoder(stock)
        if self.market_gate:
            importance = (1 + torch.tanh(self.gate(context))).repeat_interleave(self.feature_encoder.channels,dim=-1)
            stock = stock * importance[:, None]
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


class RiskAwareRanker(StockRanker):
    """Fit conditional mean and variance, then rank with an ex ante risk penalty."""
    def __init__(self, config):
        super().__init__({**config,"family":config.get("encoder","residual")})
        width = config.get("width",128)
        self.output = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2))
        self.risk_exponent = config.get("risk_exponent",0.0)
        self.risk_penalty = config.get("risk_penalty",0.0)

    def forward(self, stock, context):
        output = super().forward(stock,context).float()
        mean = output[:,0]
        log_variance = output[:,1].clamp(-6,4)
        sigma = torch.exp(0.5*log_variance).clamp_min(0.05)
        score = mean/sigma.pow(self.risk_exponent)-self.risk_penalty*sigma
        return {"mean":mean,"log_variance":log_variance,"score":score}


class FactorGaussianRanker(StockRanker):
    """Learn conditional mean and a diagonal-plus-factor covariance from past inputs."""
    def __init__(self, config):
        super().__init__({**config,"family":config.get("encoder","residual")})
        self.factor_rank = config.get("factor_rank",4)
        self.loading_bound = config.get("factor_loading_bound",2.0)
        if not isinstance(self.factor_rank,int) or self.factor_rank<0:
            raise ValueError("factor_rank must be a nonnegative integer")
        if not math.isfinite(self.loading_bound) or self.loading_bound<=0:
            raise ValueError("factor_loading_bound must be finite and positive")
        self.output = nn.Sequential(nn.LayerNorm(config.get("width",128)),
                                    nn.Linear(config.get("width",128),2+self.factor_rank))
        if self.factor_rank:
            # Zero factor weights cannot learn a covariance, because covariance is quadratic in them.
            nn.init.normal_(self.output[-1].weight[2:],std=0.02)
            nn.init.zeros_(self.output[-1].bias[2:])
        self.risk_exponent = config.get("risk_exponent",0.0)
        self.risk_penalty = config.get("risk_penalty",0.0)

    def forward(self, stock, context):
        output = super().forward(stock,context).float()
        mean = output[:,0]
        diagonal_log_variance = output[:,1].clamp(-6,4)
        loadings = self.loading_bound*torch.tanh(output[:,2:])
        variance = torch.exp(diagonal_log_variance)+loadings.square().sum(1)
        sigma = variance.sqrt().clamp_min(0.05)
        score = mean/sigma.pow(self.risk_exponent)-self.risk_penalty*sigma
        return {"mean":mean,"diagonal_log_variance":diagonal_log_variance,
                "factor_loadings":loadings,"log_variance":variance.log(),"score":score}


class QuantileAwareRanker(StockRanker):
    """Fit a mean and ordered return quantiles to penalize predicted downside."""
    def __init__(self, config):
        super().__init__({**config,"family":config.get("encoder","residual")})
        self.output = nn.Sequential(nn.LayerNorm(config.get("width",128)),nn.Linear(config.get("width",128),4))
        self.risk_penalty = config.get("risk_penalty",0.0)
        self.risk_exponent = config.get("risk_exponent",0.0)

    def forward(self, stock, context):
        output = super().forward(stock,context).float()
        mean, median = output[:,0], output[:,1]
        lower = median-F.softplus(output[:,2])
        upper = median+F.softplus(output[:,3])
        quantiles = torch.stack([lower,median,upper],dim=1)
        downside = (mean-lower).clamp_min(0.0)
        width = (upper-lower).clamp_min(0.05)
        score = mean/width.pow(self.risk_exponent)-self.risk_penalty*downside
        return {"mean":mean,"quantiles":quantiles,"score":score}


class RiskOverlayRanker(nn.Module):
    """Train an alpha ranker and apply a frozen variance penalty when scoring."""
    def __init__(self, config):
        super().__init__()
        self.alpha = StockRanker({**config,"family":config.get("encoder","residual")})
        source = config["risk_source"]
        if source["market"] != config["market"] or source["family"] not in ("risk_aware","factor_gaussian"):
            raise ValueError("Risk overlay requires a same-market conditional variance source")
        source_path = ARTIFACTS/"trials"/config_id(source)/"best.pt"
        checkpoint = torch.load(source_path,map_location="cpu",weights_only=False)
        if checkpoint["config"] != source:
            raise ValueError("Frozen risk source configuration mismatch")
        self.risk_model = make_model(source)
        self.risk_model.load_state_dict(checkpoint["model"])
        self.risk_model.requires_grad_(False)
        self.risk_model.eval()
        self.risk_penalty = config.get("risk_penalty",0.1)

    def train(self, mode=True):
        super().train(mode)
        self.risk_model.eval()
        return self

    def forward(self, stock, context):
        alpha = self.alpha(stock,context).float()
        if self.training:
            return alpha
        with torch.no_grad():
            log_variance = self.risk_model(stock,context)["log_variance"]
            standardized_risk = (log_variance-log_variance.mean())/log_variance.std(correction=0).clamp_min(0.1)
        return alpha-self.risk_penalty*standardized_risk


class BatchEnsembleRanker(nn.Module):
    """BatchEnsemble adaptation inspired by TabM, not an exact paper reproduction."""
    def __init__(self, config):
        super().__init__()
        self.context_enabled = config.get("context", True)
        self.cs_norm = config.get("cs_norm", False)
        self.feature_encoder = make_feature_encoder(config)
        members, width = config.get("members", 8), config.get("width", 128)
        dimensions = [self.feature_encoder.output_features+(76 if self.context_enabled else 0)] + [width]*config.get("depth", 3) + [1]
        self.layers = nn.ModuleList([EnsembleLinear(a, b, members) for a, b in zip(dimensions[:-1], dimensions[1:])])
        self.dropout = nn.Dropout(config.get("dropout", 0.1))

    def forward(self, stock, context):
        x = stock[:, -1]
        if self.cs_norm:
            x = (x-x.mean(0)) / x.std(0, correction=0).clamp_min(0.1)
        x = self.feature_encoder(x)
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
    if config["family"] == "master_control" and config.get("feature_encoder","identity") != "identity":
        raise ValueError("Numerical feature encoders are not implemented in MASTER control")
    if config["family"] == "risk_aware":
        return RiskAwareRanker(config)
    if config["family"] == "factor_gaussian":
        return FactorGaussianRanker(config)
    if config["family"] == "quantile_aware":
        return QuantileAwareRanker(config)
    if config["family"] == "risk_overlay":
        return RiskOverlayRanker(config)
    if config["family"] == "batch_ensemble":
        return BatchEnsembleRanker(config)
    if config["family"] == "master_control":
        return MasterControl(config)
    if config["family"] in ("residual", "temporal_mixer"):
        return StockRanker(config)
    raise ValueError(config["family"])


def uses_temporal_data(config):
    if config["family"] == "risk_overlay":
        return config.get("encoder","residual") == "temporal_mixer" or uses_temporal_data(config["risk_source"])
    return config["family"] in ("temporal_mixer","master_control") or (
        config["family"] in ("risk_aware","factor_gaussian","quantile_aware") and config.get("encoder","residual") == "temporal_mixer")


def prediction_scores(predictions):
    if isinstance(predictions,dict):
        return predictions["score"].float()
    predictions = predictions.float()
    return predictions.mean(1) if predictions.ndim == 2 else predictions


def can_pack_training_days(config):
    # All stock normalization and attention must stay inside an individual date.
    return config["family"] in ("residual","temporal_mixer","batch_ensemble","risk_aware","factor_gaussian","quantile_aware","risk_overlay") and not (
        config.get("cs_norm",False) or config.get("latent_factors",0))


def loss_by_day(predictions, labels, counts, objective, weights=None):
    """Split a packed forward pass before computing any ranking or distribution loss."""
    if sum(counts) != len(labels) or not counts or min(counts)<1:
        raise ValueError("Invalid packed day boundaries")
    weights = [1.]*len(counts) if weights is None else weights
    if len(weights) != len(counts):
        raise ValueError("Packed day weight/boundary mismatch")
    losses,offset = [],0
    for count,weight in zip(counts,weights):
        part = {key:value[offset:offset+count] for key,value in predictions.items()} if isinstance(predictions,dict) else predictions[offset:offset+count]
        losses.append(rank_loss(part,labels[offset:offset+count],objective)*weight)
        offset += count
    return torch.stack(losses)


def rank_loss(predictions, labels, objective):
    if isinstance(predictions,dict):
        if "quantiles" in predictions:
            error = labels[:,None]-predictions["quantiles"]
            levels = error.new_tensor([0.1,0.5,0.9])
            pinball = torch.maximum(levels*error,(levels-1)*error).mean()
            mse = F.mse_loss(predictions["mean"],labels)
            if objective == "quantile":
                return 0.4*mse+0.6*pinball
            ranking_objective = "corr" if objective == "quantile_rank" else objective
            return 0.3*mse+0.4*pinball+0.3*rank_loss(predictions["score"],labels,ranking_objective)
        if "factor_loadings" in predictions:
            nll = low_rank_gaussian_nll(predictions["mean"],predictions["diagonal_log_variance"],predictions["factor_loadings"],labels)
        else:
            nll = F.gaussian_nll_loss(predictions["mean"],labels,torch.exp(predictions["log_variance"]))
        if objective == "gaussian_nll":
            return nll
        ranking_objective = "corr" if objective == "gaussian_nll_rank" else objective
        return 0.7*nll+0.3*rank_loss(predictions["score"],labels,ranking_objective)
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
    if objective == "top30_pair":
        pair = topk_membership_loss(scores,labels,topk=30)
        return 0.3*mse + 0.3*(1-correlation) + 0.4*pair
    if objective == "listwise":
        distribution = F.softmax(labels / 0.7, dim=0)
        return 0.3*mse + 0.7*(-distribution*F.log_softmax(scores/0.7, dim=0)).sum()
    raise ValueError(objective)
