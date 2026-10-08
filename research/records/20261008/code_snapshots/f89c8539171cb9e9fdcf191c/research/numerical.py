"""Train-only numerical feature encodings inspired by arXiv:2203.05556.

These are local adaptations, not an exact reproduction of the paper's models.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from .common import write_json


class IdentityFeatureEncoder(nn.Identity):
    def __init__(self, features):
        super().__init__()
        self.channels = 1
        self.output_features = features


class PiecewiseLinearEncoder(nn.Module):
    """Per-feature quantile ramps, with optional raw-feature skip channels."""
    def __init__(self, features, bins=16, include_raw=True, tail_clip=2.0, minimum_width=1e-4):
        super().__init__()
        if not isinstance(bins,int) or bins<2 or minimum_width<=0:
            raise ValueError("Invalid piecewise-linear bin configuration")
        if tail_clip is not None and (not np.isfinite(tail_clip) or tail_clip<=0):
            raise ValueError("tail_clip must be positive or None")
        self.features,self.bins,self.include_raw = features,bins,include_raw
        self.channels = bins+int(include_raw)
        self.output_features = features*self.channels
        self.tail_clip,self.minimum_width = tail_clip,minimum_width
        for name in ["left","inverse_width"]:
            self.register_buffer(name,torch.zeros(features,bins))
        for name in ["active","first","last","single"]:
            self.register_buffer(name,torch.zeros(features,bins,dtype=torch.bool))
        self.register_buffer("fitted",torch.tensor(False))
        self.is_fitted = False
        self.register_load_state_dict_post_hook(self._loaded)

    def _loaded(self, module, incompatible_keys):
        self.is_fitted = bool(self.fitted.item())

    def fit(self, training_features):
        x = np.asarray(training_features,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=self.features or len(x)<2 or not np.isfinite(x).all():
            raise ValueError("Invalid training features for numerical bins")
        quantiles = np.quantile(x,np.linspace(0.,1.,self.bins+1),axis=0).T
        left,inverse = np.zeros((self.features,self.bins),np.float32),np.zeros((self.features,self.bins),np.float32)
        active,first,last,single = [np.zeros(left.shape,bool) for _ in range(4)]
        counts = []
        for feature,values in enumerate(quantiles):
            edges = [values[0]]
            for value in values[1:]:
                if value-edges[-1]>=self.minimum_width:
                    edges.append(value)
            count = len(edges)-1
            counts.append(count)
            if not count:
                continue
            edges = np.asarray(edges)
            left[feature,:count] = edges[:-1]
            inverse[feature,:count] = 1./np.diff(edges)
            active[feature,:count] = True
            first[feature,0],last[feature,count-1] = True,True
            single[feature,:count] = count==1
        for name,value in [("left",left),("inverse_width",inverse),("active",active),
                           ("first",first),("last",last),("single",single)]:
            getattr(self,name).copy_(torch.as_tensor(value,device=self.left.device))
        self.fitted.fill_(True)
        self.is_fitted = True
        return {"bins_per_feature":counts,"constant_features":sum(count==0 for count in counts),
                "buffer_sha256":hashlib.sha256(left.tobytes()+inverse.tobytes()+active.tobytes()).hexdigest()}

    def forward(self, x):
        if not self.is_fitted:
            raise ValueError("Numerical bins must be fitted on training data or loaded from a checkpoint")
        if x.shape[-1]!=self.features:
            raise ValueError("Numerical feature count mismatch")
        ramps = (x.float()[...,None]-self.left)*self.inverse_width
        encoded = ramps.clamp(0.,1.)
        encoded = torch.where(self.first,ramps.clamp_max(1.),encoded)
        encoded = torch.where(self.last,ramps.clamp_min(0.),encoded)
        encoded = torch.where(self.single,ramps,encoded)
        if self.tail_clip is not None:
            encoded = encoded.clamp(-self.tail_clip,1.+self.tail_clip)
        encoded = encoded*self.active
        if self.include_raw:
            encoded = torch.cat([x.float()[...,None],encoded],dim=-1)
        return encoded.flatten(-2)


class PeriodicFeatureEncoder(nn.Module):
    """Learn feature-specific frequencies; retain the original scalar as a skip channel."""
    def __init__(self, features, frequencies=4, initial_scale=0.05):
        super().__init__()
        if not isinstance(frequencies,int) or frequencies<1 or not np.isfinite(initial_scale) or initial_scale<=0:
            raise ValueError("Invalid periodic feature configuration")
        self.features = features
        self.channels = 1+2*frequencies
        self.output_features = features*self.channels
        self.frequencies = nn.Parameter(torch.empty(features,frequencies))
        nn.init.trunc_normal_(self.frequencies,std=initial_scale,a=-3*initial_scale,b=3*initial_scale)

    def forward(self, x):
        if x.shape[-1]!=self.features:
            raise ValueError("Numerical feature count mismatch")
        phase = x.float()[...,None]*self.frequencies*(2*np.pi)
        return torch.cat([x.float()[...,None],phase.sin(),phase.cos()],dim=-1).flatten(-2)


def make_feature_encoder(config, features=158):
    kind = config.get("feature_encoder","identity")
    if kind == "identity":
        return IdentityFeatureEncoder(features)
    if kind == "ple":
        return PiecewiseLinearEncoder(features,bins=config.get("ple_bins",16),
                    include_raw=config.get("ple_include_raw",True),tail_clip=config.get("ple_tail_clip",2.0),
                    minimum_width=config.get("ple_minimum_width",1e-4))
    if kind == "periodic":
        return PeriodicFeatureEncoder(features,frequencies=config.get("periodic_frequencies",4),
                                      initial_scale=config.get("periodic_initial_scale",0.05))
    raise ValueError(f"Unknown numerical feature encoder: {kind}")


def fit_model_feature_encoders(model, training, config, destination):
    encoders = [(name,module) for name,module in model.named_modules()
                if isinstance(module,PiecewiseLinearEncoder) and not module.is_fitted]
    if not encoders:
        return
    if training.split != "train":
        raise ValueError("Numerical bins may only be fitted on the selected training split")
    positions = training.selected_positions()
    sample_limit = config.get("ple_fit_samples",65536)
    if not isinstance(sample_limit,int) or sample_limit<2:
        raise ValueError("ple_fit_samples must be an integer >= 2")
    # Deliberately independent of the model seed: multi-seed models use the same bins.
    sample_seed = config.get("ple_fit_seed",0)
    if len(positions)>sample_limit:
        positions = np.sort(np.random.default_rng(sample_seed).choice(positions,sample_limit,replace=False))
    features = np.asarray(training.features[positions,:158],dtype=np.float32)
    fitted = {name:module.fit(features) for name,module in encoders}
    write_json(destination/"feature_encoder.json",{
        "kind":"ple","fit_split":"selected purged training rows only","fit_seed":sample_seed,
        "fit_samples":len(positions),"selected_training_samples":sum(
            int(training.boundaries[day+1]-training.boundaries[day]) for day in training.day_ids),
        "first_training_date":str(training.dates[training.day_ids].min()),
        "last_training_date":str(training.dates[training.day_ids].max()),
        "sample_positions_sha256":hashlib.sha256(positions.tobytes()).hexdigest(),
        "sample_features_sha256":hashlib.sha256(features.tobytes()).hexdigest(),
        "labels_used":False,"validation_or_test_features_used":False,"encoders":fitted,
        "application":"stock features encoded before learned market gating; context remains scalar",
        "tail_clip":config.get("ple_tail_clip",2.0),"minimum_bin_width":config.get("ple_minimum_width",1e-4)})
