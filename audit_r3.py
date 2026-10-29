"""R3: rerun saved-record checks, image-test reproduction and baseline retraining.
This does not rerun all 480 full physical trajectories; verify.py replays four.
"""
from pathlib import Path
import json, subprocess, sys, platform, hashlib
import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from train import dataset
from vision import ROOT

def main():
    subprocess.run([sys.executable, 'verify.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable, 'make_explanation_assets.py'],cwd=ROOT,check=True)
    tests=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=ROOT,check=True,capture_output=True,text=True)
    (ROOT/'results/unit_tests.txt').write_text(tests.stdout)
    saved=np.load(ROOT/'models/baseline_mlp.npz')
    x,y,_=dataset(11001,900,'standard')
    m=MLPClassifier(hidden_layer_sizes=(32,),activation='relu',solver='adam',alpha=.001,batch_size=128,learning_rate_init=.001,random_state=55001,max_iter=1,shuffle=True)
    with threadpool_limits(limits=1):
        for _ in range(60):m.partial_fit(x,y,classes=np.arange(3))
    diffs=[]
    for i,(w,b) in enumerate(zip(m.coefs_,m.intercepts_)):
        diffs += [float(np.max(np.abs(w.astype('float32')-saved[f'weight_{i}']))), float(np.max(np.abs(b.astype('float32')-saved[f'bias_{i}'])))]
    assert max(diffs)==0
    audit=json.loads((ROOT/'results/explanation_audit.json').read_text())
    out={'edition':'R3','verification':json.loads((ROOT/'results/verification.json').read_text()),
         'baseline_retraining':{'training_images':900,'epochs':60,'max_weight_difference':max(diffs),'matches':True},
         'image_tests':{k:audit['image_splits'][k]['models'] for k in ('test_standard','test_stress')},
         'counts':json.loads((ROOT/'results/counts_r3.json').read_text()),'tests':tests.stdout.strip(),
         'environment':{'python':platform.python_version(),'numpy':np.__version__},
         'scope':'Production container checks only. No real hardware or user Mac/Colab execution is claimed.',
         'unchanged_simulation':'Core model/controller/benchmark unchanged from R2.',
         'online_accounts_changed':False}
    (ROOT/'results/verification_r3.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
