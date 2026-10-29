"""One fixed training recipe. All test splits are evaluated only after fitting.

The small baseline checkpoint is retained from SIGN NAV (900 standard images,
60 epochs, 576-32-3). The upgraded MLP is trained here on 4,500 mixed images.
"""
from __future__ import annotations
import csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import sklearn, PIL
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,log_loss
from threadpoolctl import threadpool_limits
from vision import ROOT, CLASS_NAMES, render_sign, preprocess, SignClassifier

SPLITS={'train':(51001,4500,'mixed'),'validation':(52001,600,'mixed'),
        'test_standard':(53001,900,'standard'),'test_stress':(54001,900,'stress')}

def dataset(seed,n,profile):
    rng=np.random.default_rng(seed); labels=np.tile(np.arange(3),n//3);rng.shuffle(labels)
    x=np.zeros((n,576),dtype=np.float32);h=hashlib.sha256()
    for i,label in enumerate(labels):
        p=('standard' if rng.random()<.40 else 'stress') if profile=='mixed' else profile
        im=render_sign(int(label),rng,p);x[i]=preprocess(im)
        h.update(np.asarray(im).tobytes());h.update(bytes([int(label)]))
    return x,labels,h.hexdigest()

def main():
    start=time.perf_counter();ROOT.joinpath('results').mkdir(exist_ok=True)
    sets={};meta={}
    for name,(seed,n,profile) in list(SPLITS.items())[:2]:
        print('generate',name,n,flush=True);x,y,h=dataset(seed,n,profile);sets[name]=(x,y)
        meta[name]=dict(seed=seed,n=n,profile=profile,sha256=h)
    x,y=sets['train'];xv,yv=sets['validation']
    m=MLPClassifier(hidden_layer_sizes=(64,32),alpha=.005,batch_size=128,learning_rate_init=.001,
                    random_state=55002,activation='relu',solver='adam',max_iter=1,shuffle=True)
    history=[]
    with threadpool_limits(limits=1):
        for epoch in range(1,81):
            m.partial_fit(x,y,classes=np.arange(3));p=m.predict_proba(xv)
            history.append(dict(epoch=epoch,loss=float(m.loss_),validation_accuracy=float((p.argmax(1)==yv).mean()),
                                validation_log_loss=float(log_loss(yv,p,labels=[0,1,2]))))
            if epoch%20==0:print(history[-1],flush=True)
        payload=dict(n_layers=np.array(len(m.coefs_)),class_names=np.array(CLASS_NAMES),feature_size=np.array(24))
        for i,(w,b) in enumerate(zip(m.coefs_,m.intercepts_)):
            payload['weight_'+str(i)]=w.astype('float32');payload['bias_'+str(i)]=b.astype('float32')
        np.savez_compressed(ROOT/'models/robust_mlp.npz',**payload)
        strong=SignClassifier();weak=SignClassifier(ROOT/'models/baseline_mlp.npz')
        delta=float(np.max(np.abs(m.predict_proba(xv)-strong.probabilities(xv))))
        assert delta<2e-6,delta
        evals={}
        for name,(seed,n,profile) in list(SPLITS.items())[2:]:
            print('evaluate',name,n,flush=True);xx,yy,h=dataset(seed,n,profile)
            meta[name]=dict(seed=seed,n=n,profile=profile,sha256=h)
            evals[name]={}
            with (ROOT/f'results/{name}.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['i','truth','baseline_label','robust_label','robust_score'])
                bp=weak.probabilities(xx);sp=strong.probabilities(xx)
                for i in range(n):writer.writerow([i,int(yy[i]),int(bp[i].argmax()),int(sp[i].argmax()),float(sp[i].max())])
            for title,probs in [('baseline',bp),('robust',sp)]:
                pred=probs.argmax(1)
                evals[name][title]=dict(n=n,correct=int((pred==yy).sum()),accuracy=float((pred==yy).mean()),
                    confusion_matrix=confusion_matrix(yy,pred,labels=[0,1,2]).tolist())
    result=dict(model=dict(layers=[576,64,32,3],epochs=80,seed=55002,optimizer='Adam',learning_rate=.001,
                          training_rule='fixed 80 epochs; validation monitored; no test-based selection'),
                splits=meta,history=history,evaluations=evals,npz_export_max_error=delta,
                runtime_seconds=time.perf_counter()-start,
                environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,pillow=PIL.__version__))
    (ROOT/'results/vision_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(evals,indent=2),flush=True)
if __name__=='__main__':main()
