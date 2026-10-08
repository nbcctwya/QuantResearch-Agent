"""Extended native Qlib feature windows, preserving endpoint membership and labels."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.loader import DailyBatchSampler, load_dataset

from . import ARTIFACTS, EXPERIMENT_ROOT
from .common import digest_file, now, write_json


def validate_history_steps(steps):
    if isinstance(steps,bool) or not isinstance(steps,int) or steps not in (8,16,32):
        raise ValueError("history_steps must be 8, 16 or 32")
    return steps


def prepare_history(market, split, steps):
    """Cache only Alpha158 histories; original feature endpoints and label caches stay authoritative."""
    from .data import window_row_indices
    validate_history_steps(steps)
    base = ARTIFACTS/'cache'/market/split
    if steps==8:
        return base
    if split=='test' and not (ARTIFACTS/'study/selection_lock.json').exists():
        raise RuntimeError("Extended test histories require the frozen selection lock")
    base_manifest = json.loads((base/'manifest.json').read_text())
    index_hash = digest_file(base/'index.pkl')
    destination = base/f'history_{steps}'
    if (destination/'manifest.json').exists():
        metadata = json.loads((destination/'manifest.json').read_text())
        if metadata['source_sha256']!=base_manifest['source_sha256'] or metadata['endpoint_index_sha256']!=index_hash or metadata['history_steps']!=steps:
            raise ValueError("Extended history cache no longer matches the baseline endpoint cache")
        return destination
    source = EXPERIMENT_ROOT/'datasets/processed'/market/f'{split}.pkl'
    if digest_file(source)!=base_manifest['source_sha256']:
        raise ValueError("Extended history source differs from the baseline cache source")
    dataset = load_dataset(source)
    if dataset.step_len!=8 or list(dataset.group_dims.values())!=[158,13,63,1]:
        raise ValueError("Expected the original baseline sampler")
    order = DailyBatchSampler(dataset,shuffle=False).ordered_indices()
    expected_index = pd.MultiIndex.from_frame(pd.read_pickle(base/'index.pkl'))
    if not dataset.get_index()[order].equals(expected_index):
        raise ValueError("Extended history endpoints differ from the baseline cache")
    # This changes only the loaded local sampler, never the baseline pickle or configuration.
    dataset.step_len = steps
    probe = np.random.default_rng(0).choice(len(dataset),min(512,len(dataset)),replace=False)
    rows = window_row_indices(dataset,probe)
    np.testing.assert_array_equal(dataset.data_arr[rows,:158],dataset[probe][:,:,:158])
    temporary = destination.with_name(destination.name+f'.building.{os.getpid()}')
    temporary.mkdir(parents=True,exist_ok=False)
    stock = np.lib.format.open_memmap(temporary/'stock.npy',mode='w+',dtype='float32',shape=(len(order),steps,158))
    base_features = np.load(base/'features.npy',mmap_mode='r')
    for start in range(0,len(order),4096):
        end = min(start+4096,len(order))
        indices = window_row_indices(dataset,order[start:end])
        information = dataset.data_arr[indices,:158]
        if not np.isfinite(information).all():
            raise ValueError("Non-finite extended historical features")
        # Latest stock/context endpoints must not change when the history is extended.
        np.testing.assert_array_equal(information[:,-1],base_features[start:end,:158])
        stock[start:end] = information
    stock.flush()
    metadata = {'created_at':now(),'market':market,'split':split,'history_steps':steps,
                'source_sha256':base_manifest['source_sha256'],'endpoint_index_sha256':index_hash,
                'samples':len(order),'stock_shape':list(stock.shape),'stock_bytes':stock.nbytes,
                'fillna_type':dataset.fillna_type,'native_probe_windows':len(probe),
                'input_columns':'historical Alpha158 only; no labels stored in this cache',
                'endpoint_check':'all latest Alpha158 rows exactly match the original feature cache',
                'labels':'unchanged original label cache; no longer-horizon targets introduced',
                'boundary_policy':'original endpoint membership retained; trial purge and training-date selection unchanged',
                'causality_note':'inherits original learn-row label eligibility and unverified JKP revision/release timing'}
    write_json(temporary/'manifest.json',metadata)
    del stock,dataset,base_features
    gc.collect()
    temporary.rename(destination)
    print(f'Cached history {market}/{split}: {steps} steps, {metadata["samples"]} endpoints',flush=True)
    return destination
