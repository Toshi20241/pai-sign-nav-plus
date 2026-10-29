"""Fixed-setting held-out scenarios, paired policies, complete results retained.

NO successful seed filtering for the aggregate tables. Demo selection is explicit
and does not change these aggregates. Normal/variable success means completion of
all three indicated branches. Covered success means holding before unreadable
panel 2, without an earlier wrong branch. Holding is never called completion.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mission import run_episode
from vision import ROOT,SignClassifier
from policy import POLICIES,GateConfig,DecisionPolicy
from world import CONDITIONS,PracticeWorld

def wilson(k,n):
    if n==0:return [None,None]
    z=1.96;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return [max(0,c-half),min(1,c+half)]

def validation_sensitivity():
    model=SignClassifier();rows=[]
    for threshold in (.80,.90,.95,.98):
        cfg=GateConfig(threshold=threshold)
        for condition in CONDITIONS:
            successes=unsafe=holds=0;obs=0
            for seed in range(62000,62030):
                w=PracticeWorld(seed,condition);agent=DecisionPolicy('active_recheck',model,cfg)
                choices=[];stop=None
                for j in range(3):
                    agent.reset()
                    for view in range(3):
                        # Policy-only sensitivity study; no dynamics claimed here.
                        image=w._panel(j,view);d=agent.observe(image);obs+=1
                        if d['action']!='RECHECK':break
                    if d['action']=='HOLD':stop=j;holds+=1;break
                    choices.append(d['choice'])
                ev=w.evaluate(choices,stop)
                successes+=int(ev['correct_hold'] if condition=='covered' else ev['all_signs_followed'])
                unsafe+=ev['wrong_decisions']+ev['covered_entries']
            rows.append(dict(threshold=threshold,condition=condition,n=30,scenario_successes=successes,
                             unsafe_actions=unsafe,holds=holds,observations=obs))
    obj=dict(note='Validation-only policy study. Default threshold .95 was specified in GateConfig before testing; not optimized on held-out trials.',
             seeds=[62000,62029],rows=rows)
    (ROOT/'results/validation_sensitivity.json').write_text(json.dumps(obj,indent=2))
    return obj

def main(count=30,start=63000):
    if count<1:raise ValueError('count must be positive')
    begin=time.perf_counter();out=ROOT/'results/benchmark';out.mkdir(parents=True,exist_ok=True)
    cfg=GateConfig();models={'baseline_once':SignClassifier(ROOT/'models/baseline_mlp.npz')}
    robust=SignClassifier()
    spec=dict(seed_start=start,count_per_condition=count,conditions=list(CONDITIONS),policies=list(POLICIES),
              gate=asdict(cfg),pairing='same seed: same initial pose, wheel gains and per-position images across all policies',
              total_runs=count*len(CONDITIONS)*len(POLICIES),selection='every seed included; persistent condition added as a limitation test with model and gate unchanged',
              checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').glob('*.npz')})
    (out/'protocol.json').write_text(json.dumps(spec,indent=2))
    rows=[];audits=[]
    with threadpool_limits(limits=1):
        validation_sensitivity()
        for condition in CONDITIONS:
            for seed in range(start,start+count):
                for mode in POLICIES:
                    e=run_episode(seed,mode,condition,model=models.get(mode,robust),config=cfg)
                    rows.append(e['result'])
                    audits.append(dict(result=e['result'],observations=e['mission']['observations'],events=e['mission']['events']))
                if (seed-start)%10==0:print(condition,seed,'finished',len(rows),flush=True)
    (out/'trials.json').write_text(json.dumps(audits,indent=2))
    with (out/'trials.csv').open('w',newline='') as f:
        fields=[k for k,v in rows[0].items() if not isinstance(v,list)]
        wr=csv.DictWriter(f,fields,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    groups=[]
    for condition in CONDITIONS:
        for mode in POLICIES:
            rs=[r for r in rows if r['condition']==condition and r['mode']==mode];n=len(rs)
            rec=dict(condition=condition,mode=mode,n=n,
                scenario_successes=sum(r['scenario_success'] for r in rs),
                completed=sum(r['completed'] for r in rs),route_successes=sum(r['route_success'] for r in rs),
                holds=sum(r['held'] for r in rs),correct_holds=sum(r['correct_hold'] for r in rs),
                wrong_decisions=sum(r['wrong_decisions'] for r in rs),covered_entries=sum(r['covered_entries'] for r in rs),
                contacts=sum(r['contacts'] for r in rs),mean_seconds=float(np.mean([r['duration_seconds'] for r in rs])),
                mean_observations=float(np.mean([r['observations'] for r in rs])),
                mean_path_m=float(np.mean([r['path_length_m'] for r in rs])))
            rec['success_wilson95']=wilson(rec['scenario_successes'],n)
            groups.append(rec)
    summary=dict(protocol=spec,groups=groups,elapsed_seconds=time.perf_counter()-begin)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    # Same seed across modes for each demo; the first paired improvement is selected,
    # explicitly disclosed, not used to estimate overall performance.
    demo_seed=start
    for seed in range(start,start+count):
        r={r['mode']:r for r in rows if r['condition']=='variable' and r['seed']==seed}
        if not r['baseline_once']['route_success'] and r['active_recheck']['route_success']:
            demo_seed=seed;break
    for mode,name in [('baseline_once','demo_baseline'),('robust_once','demo_once'),('active_recheck','demo_active')]:
        run_episode(demo_seed,mode,'variable',ROOT/'results'/name,model=models.get(mode,robust))
    run_episode(start,'active_recheck','covered',ROOT/'results/demo_hold',model=robust)
    (ROOT/'results/demo_selection.json').write_text(json.dumps(dict(variable_seed=demo_seed,covered_seed=start,
        rule='first held-out seed where baseline fails and active succeeds; all test seeds retained in aggregate',
        role='illustrative demonstration, not an unbiased performance estimate'),indent=2))
    print(json.dumps(groups,indent=2),flush=True)
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=30);p.add_argument('--seed-start',type=int,default=63000)
    a=p.parse_args();main(a.count,a.seed_start)
