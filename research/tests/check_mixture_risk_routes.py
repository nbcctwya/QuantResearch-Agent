"""Real frozen-mixture scoring recovery and complete zero-penalty reproduction audits."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from research import ARTIFACTS
from research.common import code_fingerprint,config_id,digest_file,now,write_json


def run(config,output,bundle):
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'config.json',config)
    environment = dict(os.environ,KBS_RESEARCH_WORKSPACE_ROOT='/home/nbcctwya/kbs-workspace',
                       OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (output/'console.log').open('a') as log:
        subprocess.run([sys.executable,'-u','-m','research.train','--config',str(output/'config.json'),
                        '--out',str(output)],cwd=bundle,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((output/'result.json').read_text())


records = []
plan = json.loads((ARTIFACTS/'study/mixture_risk_smoke_plan.json').read_text())
for item in plan:
    output = Path(item['output'])
    config = json.loads(Path(item['config']).read_text())
    result = run(config,output,item['bundle'])
    assert result['smoke']
    names = ['best.pt','training_risk_statistics.npy','risk_regime.json']
    before = {name:{'sha256':digest_file(output/name),'mtime_ns':(output/name).stat().st_mtime_ns} for name in names}
    predictions = pd.read_pickle(output/'valid/predictions.pkl')
    run(config,output,item['bundle'])
    after = {name:{'sha256':digest_file(output/name),'mtime_ns':(output/name).stat().st_mtime_ns} for name in names}
    assert before==after
    pd.testing.assert_frame_equal(predictions,pd.read_pickle(output/'valid/predictions.pkl'),check_exact=True)
    records.append({'market':item['market'],'config':config,'source_bundle':item['bundle'],
                    'fitted_statistics_and_checkpoint_unchanged':True,'exact_resume_predictions':True,
                    'artifacts':after,'smoke_only':True})
    write_json(ARTIFACTS/'study/mixture_risk_gpu_checks.json',{'created_at':now(),'passed':True,
               'markets_completed':len(records),'records':records,'code':code_fingerprint()})
    print({'market':item['market'],'frozen_mixture_smoke_and_recovery':True},flush=True)

audits = []
for config in json.loads((ARTIFACTS/'study/mixture_risk_zero_audits.json').read_text()):
    output = ARTIFACTS/'audits/mixture_zero_penalty'/config_id(config)
    bundle = next(item['bundle'] for item in plan if item['market']==config['market'])
    result = run(config,output,bundle)
    alpha = ARTIFACTS/'trials'/config_id(config['alpha_source'])
    expected = pd.read_pickle(alpha/'valid/predictions.pkl')
    actual = pd.read_pickle(output/'valid/predictions.pkl')
    pd.testing.assert_frame_equal(expected,actual,check_exact=True)
    original = json.loads((alpha/'result.json').read_text())['validation']
    differences = {name:result['validation'][name]-original[name]
                   for name in ['IC','ICIR','RankIC','RankICIR','AR','STD','MDD','Sharpe','Sortino','Calmar']}
    largest = max(abs(value) for value in differences.values())
    assert largest<1e-12,(config['market'],differences)
    audits.append({'market':config['market'],'config':config,'samples':len(actual),
                   'days':actual.index.get_level_values('datetime').nunique(),
                   'exact_alpha_predictions':True,'metric_differences':differences,'max_metric_difference':largest,
                   'alpha_checkpoint_sha256':digest_file(alpha/'best.pt'),
                   'source_bundle':bundle,'scope':'complete purged validation and baseline backtest; audit excluded from search/promotion'})
    write_json(ARTIFACTS/'study/mixture_risk_zero_reproduction.json',{'created_at':now(),'records':audits,'code':code_fingerprint()})
    print({'market':config['market'],'zero_penalty_full_reproduction':True,'max_metric_difference':largest},flush=True)
