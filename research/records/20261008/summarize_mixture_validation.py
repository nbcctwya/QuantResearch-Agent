"""Export completed matched mixture controls; read no new-model test outputs."""
from __future__ import annotations

import json
import pandas as pd
import torch

from research import ARTIFACTS, ROOT
from research.common import digest_file, now, write_json
from research.models import make_model


def summarize():
    torch.set_num_threads(2)
    design = json.loads((ARTIFACTS/'study/mixture_design.json').read_text())
    records,rows = [],[]
    for item in design['comparisons']:
        folder = ARTIFACTS/'trials'/item['id']
        if not (folder/'result.json').exists():
            continue
        result = json.loads((folder/'result.json').read_text())
        assert result['status']=='complete' and not result.get('smoke')
        assert result['config']==item['config']
        config = result['config']
        diagnostic = json.loads((folder/'valid/calibration/calibration.json').read_text())
        forecasts = pd.read_pickle(folder/'valid/calibration/validation_forecasts.pkl')
        mixture = diagnostic.get('mixture')
        if mixture:
            nll = mixture['mean_negative_log_density_raw_units']
            components = config['mixture_components']
            logits = torch.tensor(forecasts[[f'component_logit_{i}' for i in range(components)]].to_numpy(),dtype=torch.float64)
            weights = torch.softmax(logits,dim=1).numpy()
            daily_weights = pd.DataFrame(weights,index=forecasts.index).groupby(level='datetime').mean()
            route_variation = {'pooled_component_weight_std':weights.std(0).tolist(),
                               'daily_mean_component_weight_std':daily_weights.std(ddof=0).tolist(),
                               'diagnostic_scope':'variation does not demonstrate semantic expert specialization'}
        else:
            observed = torch.tensor(forecasts.target.to_numpy(),dtype=torch.float64)
            normal = torch.distributions.Normal(torch.tensor(forecasts['mean'].to_numpy(),dtype=torch.float64),
                                                 torch.tensor(forecasts.sigma.to_numpy(),dtype=torch.float64))
            nll = float(-normal.log_prob(observed).mean())
            route_variation = None
        run = json.loads((folder/'run.json').read_text())
        best = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        record = {**item,'status':'complete','validation':result['validation'],
                  'seconds':result['seconds'],'resources':result.get('resources'),'best_epoch':best['epoch']+1,
                  'parameter_count':sum(p.numel() for p in make_model(config).parameters()),
                  'calibration':diagnostic,'mean_negative_log_density_raw_units':nll,
                  'routing_variation':route_variation,'code':run['code'],
                  'source_package':run['source_package'],
                  'checkpoint_sha256':digest_file(folder/'best.pt'),
                  'prediction_sha256':digest_file(folder/'valid/predictions.pkl')}
        records.append(record)
        rows.append({'variant':item['variant'],'market':item['market'],'id':item['id'],
                     'components':config.get('mixture_components',1),'gate':config.get('mixture_gate_input','none'),
                     'target_kind':config['target_kind'],'best_epoch':best['epoch']+1,
                     'parameters':record['parameter_count'],'seconds':result['seconds'],
                     'interval_80_coverage':diagnostic['interval_80_coverage'],
                     'mean_negative_log_density_raw_units':nll,**result['validation']})
    # Compare only the same target kind with its float64 one-component control.
    controls = {(r['market'],r['config']['target_kind']):r for r in records
                if r['variant'] in ('one_component','excess_one_component')}
    keys = ['IC','ICIR','RankIC','RankICIR','AR','STD','MDD','Sharpe','Sortino','Calmar']
    for record in records:
        control = controls.get((record['market'],record['config']['target_kind']))
        if control:
            record['matched_one_component'] = {'id':control['id'],'deltas':{
                key:record['validation'][key]-control['validation'][key] for key in keys},
                'negative_log_density_delta':record['mean_negative_log_density_raw_units']-control['mean_negative_log_density_raw_units']}
    destination = ROOT/'research/records/20261008'
    output = {'created_at':now(),'scope':'purged validation 2021-2022 only; smoke excluded; no new-model test outputs read',
              'planned':len(design['comparisons']),'completed':len(records),'design':design,
              'notes':['All ranking and portfolio metrics use the unchanged baseline framework.',
                       'Density diagnostics use observed validation returns and CPU float32 forecasts; densities of different target kinds are not directly comparable.',
                       'One-component float64 mixture control isolates component count; legacy Gaussian has the same initial parameters but float32 likelihood arithmetic.',
                       'No evidence here establishes exceeding any holdout baseline envelope.'],
              'cpu_checks':json.loads((ARTIFACTS/'study/mixture_checks.json').read_text()),
              'gpu_recovery':json.loads((ARTIFACTS/'study/mixture_resume_checks.json').read_text()),
              'experiments':records}
    paired_path = destination/'mixture_seed_validation_checks.json'
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        output['current_findings'] = {
            'scope':'validation only; no holdout-baseline success established',
            'matched_full_controls_complete':len(records),
            'paired_csi300_ensembles':paired['completed_ensembles'],
            'three_seed_result':'two components improve ranking and slightly reduce STD, but avg_none AR, MDD, Sharpe, Sortino and Calmar deteriorate versus one component' if paired.get('comparison') else 'paired seed evidence incomplete',
            'comparison':paired.get('comparison')}
    write_json(destination/'mixture_validation_checks.json',output)
    pd.DataFrame(rows).to_csv(destination/'mixture_validation_metrics.csv',index=False)
    print(json.dumps({'completed':len(records),'planned':output['planned'],'scope':output['scope']}))


if __name__=='__main__':
    summarize()
