from common import *
from sklearn.metrics import roc_auc_score
import json
CV=np.load('../results/color_variants.npz')
idx=gidx_paperstyle(D,20,42)
out={}
for p in ['none','lowlight','colorcast','occlusion','jpeg','blur']:
    y=CV[f'{p}__label']==1
    r={}
    for v,nm in [('a','HS (orig)'),('b','HS chroma-mask'),('c','WB + chroma-mask')]:
        s=CV[f'{p}__{v}'][:,idx].max(1)
        r[nm]=dict(auc=roc_auc_score(y,s),f1=met(y,s>=0.70)['f1'],acc=met(y,s>=0.70)['acc'])
    out[p]=r
    print(p,{k:{kk:round(vv,3) for kk,vv in v.items()} for k,v in r.items()})
# background shortcut
bg=CV['none__bg']; y=D['label']==1
print('bg frac genuine mean',bg[y].mean(),'cf',bg[~y].mean(),'AUC bg alone',roc_auc_score(y,bg),'gallery bg',CV['gallery_bg'].mean())
json.dump(out,open('colorvar.json','w'),default=float,indent=1)
