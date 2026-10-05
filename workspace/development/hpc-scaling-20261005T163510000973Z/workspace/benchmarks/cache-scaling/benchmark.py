"""Bounded CPU profile: 16 real episodes per corpus, no GPU or training."""
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from pi_jepa.data import TrainDataset
from pi_jepa.initial_conditions import FixedInitialConditions
from pi_jepa.losses import simulate_window
from pi_jepa.train import batch_from
from pi_jepa.training_cache import prepare_learning_cache, prepare_fixed_target_cache, FixedTargetCache

torch.set_num_threads(2)
base=Path('workspace/benchmarks/cache-scaling').resolve()
production=Path('/home/rvalperga/pi-jepa/workspace/data')
results={'torch_threads':2,'device':'cpu','protocol':'16 deterministic real training episodes per corpus; local derived copies, warm OS file cache, unchanged actual RGB/forces/theta/reset values; medians of three repetitions unless specified','datasets':{}}

def timed(fn,repeats=3):
    seconds=[]
    for _ in range(repeats):
        start=time.perf_counter(); value=fn();seconds.append(time.perf_counter()-start)
    return statistics.median(seconds),value

for kind in ['passive','controlled']:
    original=json.loads((production/kind/'manifest.json').read_text())
    selected=np.random.default_rng(913).choice(len(original['train']),16,replace=False)
    manifest=dict(original);manifest['train']=[original['train'][i] for i in selected];manifest['validation']=[]
    root=base/'data'/kind;root.mkdir(parents=True,exist_ok=True)
    initial=[]
    for entry in manifest['train']:
        dest=root/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(production/kind/entry['path'],dest)
        with np.load(production/kind/entry['truth'],allow_pickle=False) as file:
            initial.append(file['exact_initial_state'])
    (root/'manifest.json').write_text(json.dumps(manifest))
    table=FixedInitialConditions(np.stack(initial),[e['reset_mode'] for e in manifest['train']])
    metadata={'mode':'true_fixed','split':'train','source_sha256':hashlib.sha256(table.states.numpy().tobytes()).hexdigest()}
    elapsed,path=timed(lambda:prepare_learning_cache(root,base/'cache',splits=('train',)),repeats=1)
    target_prep,targetpath=timed(lambda:prepare_fixed_target_cache(root,base/'cache',table,metadata,batch_size=16),repeats=1)
    cache=FixedTargetCache(root,base/'cache',table,metadata)
    plain=TrainDataset(root,cache_size=0)
    mapped=TrainDataset(root,cache_size=0,cache_root=base/'cache')
    stats={'original_training_indices':selected.tolist(),'learning_cache_prepare_s':elapsed,'target_cache_prepare_s':target_prep,'uncompressed_learning_bytes':sum(p.stat().st_size for p in path.rglob('*.npy')),'fixed_target_bytes':sum(p.stat().st_size for p in targetpath.glob('*.npy')),'batches':{}}
    for n in [8,16]:
        ids=torch.arange(n)
        starts=[int(np.random.default_rng(442+i).integers(0,manifest['train'][i]['valid_length']-65+1)) for i in range(n)]
        raw_seconds,raw=timed(lambda:batch_from(plain,ids,'cpu',starts=starts))
        cache_seconds,cached=timed(lambda:batch_from(mapped,ids,'cpu',starts=starts))
        assert all(torch.equal(raw[k],cached[k]) for k in raw)
        solver_seconds,target=timed(lambda:simulate_window(table(ids),raw['theta'],raw['prefix_forces'],raw['raw_endpoints']))
        lookup_seconds,lookedup=timed(lambda:cache.gather(ids,raw['raw_endpoints']),repeats=101)
        maxdiff=float((target-lookedup).abs().max())
        stats['batches'][n]={'npz_load_and_crop_s':raw_seconds,'mmap_load_and_crop_s':cache_seconds,'load_speedup':raw_seconds/cache_seconds,'float64_solver_s':solver_seconds,'fixed_target_lookup_s':lookup_seconds,'target_speedup':solver_seconds/lookup_seconds,'target_bitwise_equal':torch.equal(target,lookedup),'target_max_abs_difference':maxdiff,'max_raw_endpoint':int(raw['raw_endpoints'].max())}
    results['datasets'][kind]=stats
    print(json.dumps({kind:stats},indent=2),flush=True)
(base/'results.json').write_text(json.dumps(results,indent=2)+'\n')
