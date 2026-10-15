from common import *
from sklearn.metrics import roc_auc_score
y=D['label']==1; n=68
def casc(d,idx): o,c,t=mods(d,idx); return (o.astype(int)+c+t)>=2
# seed variance k=20
R={}
for k in [1,2,5,10,20,40,68]:
    rows=[]
    for s in range(20 if k<68 else 1):
        idx=random.Random(1000+s).sample(range(n),k)
        o,c,t=mods(D,idx); v=o.astype(int)+c+t
        rows.append([met(y,v>=2)['f1'],met(y,v>=2)['acc'],met(y,v>=2)['rc'],met(y,v>=2)['pr'],met(y,c)['f1'],met(y,c)['acc'],met(y,v>=1)['f1'],met(y,o)['f1']])
    a=np.array(rows); R[k]=a
    print(k,' '.join(f'{m:.3f}±{s:.3f}' for m,s in zip(a.mean(0),a.std(0))))
