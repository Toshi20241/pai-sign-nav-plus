"""Audit saved sensor/policy traces and selected closed-loop simulation replays."""
from __future__ import annotations
import hashlib,inspect,json,math,time
import numpy as np
from threadpoolctl import threadpool_limits
from vision import ROOT,SignClassifier
from policy import DecisionPolicy,GateConfig
from world import PracticeWorld
from mission import run_episode

def main():
    started=time.perf_counter();audits=json.loads((ROOT/'results/benchmark/trials.json').read_text())
    models={'baseline_once':SignClassifier(ROOT/'models/baseline_mlp.npz')};strong=SignClassifier()
    nobs=0;tested_images=set();sensor_identity={}
    with threadpool_limits(limits=1):
        for rec in audits:
            result=rec['result'];world=PracticeWorld(result['seed'],result['condition'])
            model=models.get(result['mode'],strong);agent=DecisionPolicy(result['mode'],model,GateConfig())
            last_j=-1;choices=[]
            for o in rec['observations']:
                if o['junction']!=last_j:agent.reset();last_j=o['junction']
                image,view=world.capture_sign(o['junction'],o['robot_pose'])
                h=hashlib.sha256(image.tobytes()).hexdigest()
                assert h==o['image_sha256'] and view==o['view']
                key=(result['seed'],result['condition'],o['junction'],view)
                if key in sensor_identity:assert sensor_identity[key]==h
                sensor_identity[key]=h;tested_images.add(key)
                d=agent.observe(image)
                assert d['action']==o['action'] and d['choice']==o['choice']
                assert np.allclose(d['prediction']['probabilities'],o['prediction']['probabilities'],rtol=0,atol=1e-6)
                if d['action']=='GO':choices.append(d['choice'])
                nobs+=1
            assert choices==result['choices']
            ev=world.evaluate(choices,result['stopped_at'])
            for k,v in ev.items():assert v==result[k],(k,v,result[k])
            assert result['max_wheel_acceleration']<=32.000001
            assert not (result['held'] and result['completed'])
    table=json.loads((ROOT/'results/benchmark/summary.json').read_text())
    for group in table['groups']:
        rs=[a['result'] for a in audits if a['result']['condition']==group['condition'] and a['result']['mode']==group['mode']]
        assert len(rs)==group['n']
        pairs={'scenario_successes':'scenario_success','completed':'completed','route_successes':'route_success',
               'holds':'held','correct_holds':'correct_hold','wrong_decisions':'wrong_decisions',
               'covered_entries':'covered_entries','contacts':'contacts'}
        for target,source in pairs.items():assert group[target]==sum(r[source] for r in rs)
        assert abs(group['mean_seconds']-float(np.mean([r['duration_seconds'] for r in rs])))<1e-9
    replayed=[]
    for sub in ('demo_baseline','demo_once','demo_active','demo_hold'):
        saved=json.loads((ROOT/'results'/sub/'mission.json').read_text());r=saved['result']
        e=run_episode(r['seed'],r['mode'],r['condition'],record=True)
        for k in ('completed','held','choices','contacts','scenario_success','duration_seconds'):
            assert e['result'][k]==r[k],(sub,k)
        assert np.allclose(e['result']['final_position'],r['final_position'],atol=1e-9)
        replayed.append(sub)
    policy_source=inspect.getsource(DecisionPolicy)
    assert '_labels' not in policy_source and 'PracticeWorld' not in policy_source
    summary=dict(passed=True,missions_audited=len(audits),observations_reinferred=nobs,
                 unique_sensor_views=len(tested_images),matched_policy_sensor_identity=True,
                 outcome_counters_recomputed=True,aggregate_tables_recomputed=True,closed_loop_replays=replayed,
                 bounded_wheel_acceleration=True,policy_interface='image pixels only; no world or target label',
                 seconds=time.perf_counter()-started)
    (ROOT/'results/verification.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
