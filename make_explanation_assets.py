"""Rebuild factual illustration assets from the existing models and experiment logs.
No model, controller, or benchmark settings are changed by this script.
"""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from vision import ROOT, CLASS_NAMES, render_sign, preprocess, SignClassifier

ASSETS = ROOT / 'presentation' / 'assets'
NAMES = ['左', '直進', '右']

def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    metrics=load_json(ROOT/'results/vision_metrics.json')
    models={'baseline':SignClassifier(ROOT/'models/baseline_mlp.npz'),'robust':SignClassifier()}
    audit={'unchanged_experiment':True,'model_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').glob('*.npz')},'image_splits':{},'demo':{}}
    samples={}
    splits=[('baseline_train',11001,900,'standard'),('robust_train',51001,4500,'mixed'),
            ('validation',52001,600,'mixed'),('test_standard',53001,900,'standard'),('test_stress',54001,900,'stress')]
    with threadpool_limits(limits=1):
        for title,seed,n,profile in splits:
            rng=np.random.default_rng(seed); labels=np.tile(np.arange(3),n//3);rng.shuffle(labels)
            x=np.zeros((n,576),np.float32);h=hashlib.sha256();seen={}; counts={'standard':0,'stress':0}
            for i,label in enumerate(labels):
                actual=('standard' if rng.random()<.40 else 'stress') if profile=='mixed' else profile
                counts[actual]+=1
                im=render_sign(int(label),rng,actual);x[i]=preprocess(im)
                h.update(np.asarray(im).tobytes());h.update(bytes([int(label)]))
                if int(label) not in seen:
                    path=ASSETS/f'{title}_{int(label)}.png';im.save(path)
                    seen[int(label)]={'i':i,'truth':int(label),'profile':actual,'file':path.name}
            key='train' if title=='robust_train' else title
            if key in metrics['splits']:
                assert metrics['splits'][key]['sha256']==h.hexdigest(), (title,'dataset differs')
            result={'seed':seed,'n':n,'per_direction':np.bincount(labels,minlength=3).tolist(),'profiles':counts,'sha256':h.hexdigest(),'samples':seen}
            if title.startswith('test_'):
                result['models']={}
                saved=list(csv.DictReader((ROOT/f'results/{title}.csv').open()))
                for modelname,model in models.items():
                    probs=model.probabilities(x);preds=probs.argmax(axis=1)
                    correct=int((preds==labels).sum())
                    assert correct==metrics['evaluations'][title][modelname]['correct']
                    assert [int(r[f'{modelname}_label']) for r in saved]==preds.tolist()
                    result['models'][modelname]={'correct':correct,'n':n}
            audit['image_splits'][title]=result
            print(title,n,'verified',flush=True)
    for lab in range(3):
        render_sign(lab,np.random.default_rng(0),'canonical').save(ASSETS/f'reference_{lab}.png')
    for name in ('demo_baseline','demo_once','demo_active','demo_hold'):
        m=load_json(ROOT/'results'/name/'mission.json')
        audit['demo'][name]={'result':m['result'],'observations':[]}
        for o in m['observations']:
            im=Image.open(ROOT/'results'/name/o['image']).convert('RGB')
            assert hashlib.sha256(im.tobytes()).hexdigest()==o['image_sha256']
            basename=f'{name}_j{o["junction"]+1}_v{o["view"]+1}.png';im.save(ASSETS/basename)
            audit['demo'][name]['observations'].append({**o,'asset':basename})
    assert audit['demo']['demo_baseline']['result']['true_labels']==[0,2,0]
    assert audit['demo']['demo_baseline']['result']['choices']==[0,2,1]
    assert audit['demo']['demo_once']['result']['choices']==[0,2,0]
    assert audit['demo']['demo_active']['result']['choices']==[0,2,0]
    for j in range(3):
        hashes=[next(o['image_sha256'] for o in audit['demo'][name]['observations'] if o['junction']==j and o['view']==0) for name in ('demo_baseline','demo_once','demo_active')]
        assert len(set(hashes))==1
    audit['first_view_identical_across_three_demos']=True
    audit['important_interpretation']='demo_once already succeeds at seed 63000; A-vs-D video is not a pure re-observation ablation.'
    (ROOT/'results/explanation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return audit

if __name__=='__main__': main()
