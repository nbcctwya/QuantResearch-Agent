"""Cache exact Qlib windows in date order, excluding labels from model inputs."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from datasets.loader import DailyBatchSampler, load_dataset

from . import ARTIFACTS, EXPERIMENT_ROOT
from .common import digest_file, now, write_json


def window_row_indices(sampler, positions):
    rows, columns = sampler.idx_map[np.asarray(positions)].T
    history_rows = rows[:, None] + np.arange(1 - sampler.step_len, 1)[None, :]
    indices = sampler.idx_arr[np.maximum(history_rows, 0), columns[:, None]].copy()
    indices[history_rows < 0] = np.nan
    if sampler.fillna_type in ("ffill", "ffill+bfill"):
        positions_in_window = np.arange(sampler.step_len)[None, :]
        forward = np.maximum.accumulate(np.where(np.isnan(indices), 0, positions_in_window), axis=1)
        indices = np.take_along_axis(indices, forward, axis=1)
    if sampler.fillna_type == "ffill+bfill":
        reversed_indices = indices[:, ::-1]
        backward = np.maximum.accumulate(np.where(np.isnan(reversed_indices), 0, positions_in_window), axis=1)
        indices = np.take_along_axis(reversed_indices, backward, axis=1)[:, ::-1]
    return np.nan_to_num(indices, nan=sampler.nan_idx).astype(np.int64)


def prepare_cache(market, split):
    destination = ARTIFACTS / "cache" / market / split
    source = EXPERIMENT_ROOT / "datasets" / "processed" / market / f"{split}.pkl"
    configuration = EXPERIMENT_ROOT / "configs" / "data" / f"{market}_dataset.yaml"
    saved_config = source.with_name("config.yaml")
    if yaml.safe_load(configuration.read_text()) != yaml.safe_load(saved_config.read_text()):
        raise ValueError(f"Dataset configuration mismatch: {market}")
    if (destination / "manifest.json").exists():
        import json
        metadata = json.loads((destination / "manifest.json").read_text())
        if metadata["source_sha256"] != digest_file(source):
            raise ValueError("Cached dataset source has changed")
        return destination
    temporary = destination.with_name(destination.name + f".building.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(source)
    if dataset.step_len != 8 or list(dataset.group_dims.values()) != [158, 13, 63, 1]:
        raise ValueError("Expected the baseline-compatible 8 x 235 layout")
    sampler = DailyBatchSampler(dataset, shuffle=False)
    order = sampler.ordered_indices()
    index = dataset.get_index()[order]
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Invalid date/stock alignment")
    probe = np.random.default_rng(0).choice(len(dataset), min(512, len(dataset)), replace=False)
    # Check padding, filling and the whole window against Qlib, before caching.
    np.testing.assert_array_equal(dataset.data_arr[window_row_indices(dataset, probe)], dataset[probe])
    stock = np.lib.format.open_memmap(temporary / "stock.npy", mode="w+", dtype="float32", shape=(len(order), 8, 158))
    features = np.lib.format.open_memmap(temporary / "features.npy", mode="w+", dtype="float32", shape=(len(order), 234))
    labels = np.lib.format.open_memmap(temporary / "label.npy", mode="w+", dtype="float32", shape=(len(order),))
    for start in range(0, len(order), 8192):
        end = min(start + 8192, len(order))
        row_indices = window_row_indices(dataset, order[start:end])
        information = dataset.data_arr[row_indices, :158]
        last = dataset.data_arr[row_indices[:, -1]]
        if not np.isfinite(information).all() or not np.isfinite(last[:, :234]).all():
            raise ValueError("Non-finite model inputs")
        stock[start:end] = information
        features[start:end] = last[:, :234]
        labels[start:end] = last[:, 234]
    for array in (stock, features, labels):
        array.flush()
    index.to_frame(index=False).to_pickle(temporary / "index.pkl")
    dates = index.get_level_values("datetime").to_numpy()
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(index)].astype(np.int64)
    np.save(temporary / "boundaries.npy", boundaries)
    np.save(temporary / "dates.npy", dates[boundaries[:-1]].astype("datetime64[ns]"))
    metadata = {
        "created_at": now(), "market": market, "split": split,
        "source_sha256": digest_file(source), "config_sha256": digest_file(configuration),
        "samples": len(index), "days": len(boundaries) - 1,
        "stock_shape": list(stock.shape), "feature_shape": list(features.shape),
        "input_columns": "Alpha158 plus prior and market; label column 234 is excluded",
        "alignment": "datetime/instrument sorted; all windows checked against Qlib on 512 positions",
        "label_kind": "raw return" if split == "test" else "CSRankNorm",
        "original_boundary_policy": "source samplers retained; training/selection purge is applied by the runner",
    }
    write_json(temporary / "manifest.json", metadata)
    del stock, features, labels, dataset
    gc.collect()
    temporary.rename(destination)
    print(f"Cached {market}/{split}: {metadata['samples']} samples, {metadata['days']} days", flush=True)
    return destination


class DailyData:
    def __init__(self, market, split, purge_days=5, train_start=None, limit_days=None, temporal=False, device="cpu"):
        import torch
        path = ARTIFACTS / "cache" / market / split
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"Prepare cache first: {path}")
        self.market, self.split = market, split
        self.temporal = temporal
        self.features = np.load(path / "features.npy", mmap_mode="r")
        self.stock = np.load(path / "stock.npy", mmap_mode="r") if temporal else None
        self.labels = np.load(path / "label.npy", mmap_mode="r")
        self.boundaries = np.load(path / "boundaries.npy")
        self.dates = np.load(path / "dates.npy")
        frame = pd.read_pickle(path / "index.pkl")
        self.index = pd.MultiIndex.from_frame(frame)
        self.day_ids = np.arange(len(self.dates))
        if split in ("train", "valid") and purge_days:
            self.day_ids = self.day_ids[:-purge_days]
        if split == "train" and train_start:
            self.day_ids = self.day_ids[self.dates[self.day_ids] >= np.datetime64(train_start)]
        if limit_days is not None:
            self.day_ids = self.day_ids[:limit_days]
        if not len(self.day_ids):
            raise ValueError("Empty selected date range")
        self.device = torch.device(device)
        self.gpu_features = self.gpu_stock = self.gpu_labels = None

    def preload(self, budget_bytes):
        import torch
        required = self.features.nbytes + self.labels.nbytes
        if self.temporal:
            required += self.stock.nbytes
        if self.device.type != "cuda" or required > budget_bytes:
            return 0
        def copy_chunks(array):
            tensor = torch.empty(array.shape, dtype=torch.float32, device=self.device)
            for start in range(0, len(array), 16384):
                tensor[start:start+16384] = torch.from_numpy(np.array(array[start:start+16384], copy=True)).to(self.device)
            return tensor
        try:
            self.gpu_features = copy_chunks(self.features)
            self.gpu_labels = copy_chunks(self.labels)
            if self.temporal:
                self.gpu_stock = copy_chunks(self.stock)
        except torch.cuda.OutOfMemoryError:
            self.gpu_features = self.gpu_stock = self.gpu_labels = None
            torch.cuda.empty_cache()
            return 0
        return required

    def inputs(self, day):
        import torch
        start, end = map(int, self.boundaries[day:day+2])
        def tensor(array, cached):
            if cached is not None:
                return cached[start:end]
            return torch.from_numpy(np.array(array[start:end], copy=True)).to(self.device)
        features = tensor(self.features, self.gpu_features)
        stock = tensor(self.stock, self.gpu_stock) if self.temporal else features[:, None, :158]
        return stock, features[:, 158:234]

    def batch(self, day):
        import torch
        stock, context = self.inputs(day)
        start, end = map(int, self.boundaries[day:day+2])
        labels = self.gpu_labels[start:end] if self.gpu_labels is not None else torch.from_numpy(
            np.array(self.labels[start:end], copy=True)).to(self.device)
        return stock, context, labels

    def selected_positions(self):
        return np.concatenate([np.arange(self.boundaries[d], self.boundaries[d+1]) for d in self.day_ids])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, choices=["csi300", "sp500"])
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()
    for split in args.splits:
        prepare_cache(args.market, split)


if __name__ == "__main__":
    main()
