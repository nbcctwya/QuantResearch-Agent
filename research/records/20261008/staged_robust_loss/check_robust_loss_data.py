"""Real complete training targets and independent robust gradients; no forecasts."""
import json
from pathlib import Path

import numpy as np
import torch

from research import ARTIFACTS
from research.common import code_fingerprint,digest_file,now,write_json
from research.data import DailyData
from research.protocol import raw_labels
from research.robust_loss import robust_regression_loss
from research.train import standardized_return_targets


def states(paths):
    return {str(path):{"sha256":digest_file(path),"mtime_ns":path.stat().st_mtime_ns} for path in paths}


def main():
    study=ARTIFACTS/"study";design=json.loads((study/"robust_loss_design.json").read_text())
    assert code_fingerprint()==design["code"]
    assert Path(__import__("research").__file__).parent==Path(design["bundle"])/"research"
    records=[]
    for market in ["csi300","sp500"]:
        data=DailyData(market,"train",purge_days=5);index=data.index[data.selected_positions()]
        paths=[ARTIFACTS/"cache"/market/"train"/name for name in ["features.npy","label.npy","index.pkl","manifest.json"]]
        before=states(paths);assert index[-1][0].year==2020 and index[0][0].year==2009
        raw=raw_labels(market,index).to_numpy(dtype=np.float32);raw_saved=raw.copy();cases=[]
        for kind in ["raw_standardized","raw_excess_standardized"]:
            labels,scale=standardized_return_targets(raw,index,kind)
            reference=next(row for row in design["existing_controls"] if row["config"]["market"]==market and row["config"]["target_kind"]==kind)
            folder=Path(reference["directory"])
            assert all(digest_file(folder/name)==value for name,value in reference["files"].items())
            saved_transform=json.loads((folder/"target_transform.json").read_text());assert scale==saved_transform["scale"]
            targets=torch.from_numpy(labels.astype(np.float64));unchanged=labels.copy()
            for delta in design["threshold_grid"]:
                scores=torch.zeros(len(labels),dtype=torch.float64,requires_grad=True)
                value=robust_regression_loss(scores,targets,delta)
                error=-labels.astype(np.float64);absolute=np.abs(error)
                expected=float(np.where(absolute<=delta,error*error,2*delta*absolute-delta*delta).mean())
                np.testing.assert_allclose(value.item(),expected,rtol=0,atol=1e-12)
                gradient,=torch.autograd.grad(value,scores)
                expected_gradient=2*np.clip(error,-delta,delta)/len(labels)
                np.testing.assert_allclose(gradient.numpy(),expected_gradient,rtol=0,atol=1e-16)
                np.testing.assert_array_equal(labels,unchanged)
                cases.append({"target_kind":kind,"delta":delta,"scale":scale,
                    "independent_loss_error":float(value.item()-expected),
                    "max_independent_gradient_error":float(np.max(np.abs(gradient.numpy()-expected_gradient))),
                    "fraction_outside_quadratic_region_at_zero_forecast":float(np.mean(absolute>delta)),
                    "target_sha256":__import__("hashlib").sha256(labels.tobytes()).hexdigest(),
                    "target_and_scale_match_existing_full_control":True})
        np.testing.assert_array_equal(raw,raw_saved);assert before==states(paths)
        records.append({"market":market,"rows":len(index),"last_training_forecast":str(index[-1][0]),
            "cases":cases,"input_files_unchanged":before})
        write_json(study/"robust_loss_data_checks.json",{"created_at":now(),"passed":len(records)==2,
            "scope":"complete selected training returns only; loss and analytic gradients at zero are preflight probes, not model predictions or performance results; no validation/test query",
            "markets_completed":len(records),"records":records,"code":code_fingerprint(),"harness_sha256":digest_file(Path(__file__))})
        print({"market":market,"rows":len(index),"independent_full_target_loss_and_gradient_cases":6,"source_files_unchanged":True},flush=True)


if __name__=="__main__":main()
