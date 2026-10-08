"""Interrupt after one saved epoch, then verify exact continuation on both markets."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pandas as pd
import torch

from research import ARTIFACTS
from research.common import code_fingerprint, digest_file, now, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--plan',type=Path,default=ARTIFACTS/'study/mixture_smoke_plan.json')
parser.add_argument('--record',type=Path,default=ARTIFACTS/'study/mixture_resume_checks.json')
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
records = []
environment = dict(os.environ, KBS_RESEARCH_WORKSPACE_ROOT='/home/nbcctwya/kbs-workspace',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
for item in plan:
    original = Path(item['output'])
    recovered = original.with_name(original.name+'_interrupted_'+uuid.uuid4().hex[:8])
    recovered.mkdir(exist_ok=False)
    config = json.loads(Path(item['config']).read_text())
    write_json(recovered/'config.json', config)
    # Raise on the first group of epoch two, after the original epoch-one checkpoint.
    harness = '''
import json, sys
from pathlib import Path
from research import train
original_loss = train.training_group_losses
groups = 0
def stop_after_saved_epoch(*args, **kwargs):
    global groups
    groups += 1
    if groups > 8:
        raise RuntimeError('intentional mixture recovery probe')
    return original_loss(*args, **kwargs)
train.training_group_losses = stop_after_saved_epoch
try:
    train.run_training(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
except RuntimeError as error:
    if str(error) != 'intentional mixture recovery probe':
        raise
    print(str(error), flush=True)
    raise SystemExit(42)
'''
    with (recovered/'recovery_probe.log').open('w') as stream:
        first = subprocess.run([sys.executable,'-u','-c',harness,str(recovered/'config.json'),str(recovered)],
                               cwd=item['bundle'],env=environment,stdout=stream,stderr=subprocess.STDOUT)
        assert first.returncode==42, (item['market'],first.returncode)
        interrupted = torch.load(recovered/'last.pt',map_location='cpu',weights_only=False)
        assert interrupted['epoch']==0
        subprocess.run([sys.executable,'-u','-m','research.train','--config',str(recovered/'config.json'),
                        '--out',str(recovered)],cwd=item['bundle'],env=environment,
                        stdout=stream,stderr=subprocess.STDOUT,check=True)
    expected_checkpoint = torch.load(original/'last.pt',map_location='cpu',weights_only=False)
    actual_checkpoint = torch.load(recovered/'last.pt',map_location='cpu',weights_only=False)
    for key,value in expected_checkpoint['model'].items():
        torch.testing.assert_close(value,actual_checkpoint['model'][key],rtol=0,atol=0)
    expected_best = torch.load(original/'best.pt',map_location='cpu',weights_only=False)
    actual_best = torch.load(recovered/'best.pt',map_location='cpu',weights_only=False)
    assert expected_best['epoch']==actual_best['epoch']
    for key,value in expected_best['model'].items():
        torch.testing.assert_close(value,actual_best['model'][key],rtol=0,atol=0)
    expected_epochs = [json.loads(line) for line in (original/'epochs.jsonl').read_text().splitlines()]
    actual_epochs = [json.loads(line) for line in (recovered/'epochs.jsonl').read_text().splitlines()]
    for expected,actual in zip(expected_epochs,actual_epochs):
        for key in ['epoch','train_loss','valid_selection_score','best_selection_score','valid','optimizer_steps']:
            assert expected[key]==actual[key], (key,expected[key],actual[key])
    assert len(expected_epochs)==len(actual_epochs)==3
    pd.testing.assert_frame_equal(pd.read_pickle(original/'valid/predictions.pkl'),
                                  pd.read_pickle(recovered/'valid/predictions.pkl'),check_exact=True)
    before = {name:{'sha256':digest_file(recovered/name),'mtime_ns':(recovered/name).stat().st_mtime_ns}
              for name in ['best.pt','last.pt','epochs.jsonl']}
    with (recovered/'finished_resume.log').open('w') as stream:
        subprocess.run([sys.executable,'-u','-m','research.train','--config',str(recovered/'config.json'),
                        '--out',str(recovered)],cwd=item['bundle'],env=environment,
                        stdout=stream,stderr=subprocess.STDOUT,check=True)
    after = {name:{'sha256':digest_file(recovered/name),'mtime_ns':(recovered/name).stat().st_mtime_ns} for name in before}
    assert before==after
    record = {'market':item['market'],'config':config,'source_bundle':item['bundle'],
              'interrupted_epoch':1,'final_epoch':3,'exact_model_parameters':True,'exact_best_checkpoint_parameters':True,
              'exact_losses_and_validation_predictions':True,'completed_resume_artifacts_unchanged':before==after,
              'artifacts':after,'smoke_only':True,'checkpoint_before_best_comparison':str(original/'best.pt'),'recovered_directory':str(recovered)}
    records.append(record)
    write_json(args.record,{'created_at':now(),'passed':True,
               'markets_completed':len(records),'records':records,'code':code_fingerprint(),
               'scope':'GPU smoke continuation and exhausted-epoch recovery; excluded from full validation counts'})
    print(json.dumps({'market':item['market'],'history_steps':config.get('history_steps'),
                      'exact_recovery':True,'completed_resume_unchanged':True}),flush=True)
