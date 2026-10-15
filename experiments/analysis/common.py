import numpy as np, random
D=np.load('../results/scores_none.npz',allow_pickle=True)
def load(n): return np.load(f'../results/scores_{n}.npz',allow_pickle=True)
def gidx(d,k,seed=42):
    n=len(d['gallery']); return sorted(random.Random(seed).sample(range(n),k)) if k<n else list(range(n))
def gidx_paperstyle(d,k,seed=42):
    n=len(d['gallery']); return random.Random(seed).sample(range(n),min(k,n))
def mods(d,idx,to=15,tc=0.70,tt=0.80):
    o=d['orb'][:,idx].max(1)>=to; c=d['col'][:,idx].max(1)>=tc; t=d['ocr']>=tt
    return o,c,t
def met(y,p):
    y=np.asarray(y).astype(bool);p=np.asarray(p).astype(bool)
    tp=(y&p).sum();tn=(~y&~p).sum();fp=(~y&p).sum();fn=(y&~p).sum()
    acc=(tp+tn)/len(y);pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
    f1=2*pr*rc/(pr+rc) if pr+rc else 0
    return dict(tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn),acc=acc,pr=pr,rc=rc,f1=f1)
