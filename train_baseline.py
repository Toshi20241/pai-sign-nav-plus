"""Optional reproduction of the retained small SIGN NAV baseline checkpoint."""
import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from vision import ROOT,CLASS_NAMES
from train import dataset

def main():
    x,y,_=dataset(11001,900,'standard')
    m=MLPClassifier(hidden_layer_sizes=(32,),activation='relu',solver='adam',alpha=.001,batch_size=128,
                    learning_rate_init=.001,random_state=55001,max_iter=1,shuffle=True)
    with threadpool_limits(limits=1):
        for _ in range(60):m.partial_fit(x,y,classes=np.arange(3))
    payload=dict(n_layers=np.array(len(m.coefs_)),class_names=np.array(CLASS_NAMES),feature_size=np.array(24),training_epochs=np.array(60))
    for i,(w,b) in enumerate(zip(m.coefs_,m.intercepts_)):payload[f'weight_{i}']=w.astype('float32');payload[f'bias_{i}']=b.astype('float32')
    np.savez_compressed(ROOT/'models/baseline_mlp.npz',**payload)
    print('Baseline checkpoint regenerated; rerun experiments to update checksums.')
if __name__=='__main__':main()
