from common import *
from sklearn.metrics import roc_auc_score
import json
idx=gidx_paperstyle(D,20,42)
pos={p:i for i,p in enumerate(D['path'])}
out={}
for p in ['none','blur','jpeg','lowlight','lowres','rotate','perspective','occlusion','colorcast']:
    d=load(p) if p!='none' else None
    if p=='none':
        s=load('blur'); sel=[pos[x] for x in s['path']]
        sub={k:(D[k][sel] if D[k].ndim and D[k].shape[0]==1600 else D[k]) for k in D.files}
        d=sub
    y=d['label']==1
    o,c,t=mods(d,idx); v=o.astype(int)+c+t
    mc=met(y,v>=2); mcol=met(y,c); mor=met(y,v>=1)
    cnn=d['cnn'][:,idx].max(1)
    r=dict(casc=mc,color=mcol,orc=mor,logo_rc=met(y,o)['rc'],ocr_hits=int((t).sum()),auc_color=roc_auc_score(y,d['col'][:,idx].max(1)),auc_cnn=roc_auc_score(y,cnn),auc_orb=roc_auc_score(y,d['orb'][:,idx].max(1)),
           pass_rate_g=float((v>=2)[y].mean()),pass_rate_c=float((v>=2)[~y].mean()))
    out[p]=r
    print(f"{p:12s} casc acc={mc['acc']:.3f} P={mc['pr']:.3f} R={mc['rc']:.3f} F1={mc['f1']:.3f} | color acc={mcol['acc']:.3f} F1={mcol['f1']:.3f} AUC={r['auc_color']:.3f} | orbAUC={r['auc_orb']:.3f} cnnAUC={r['auc_cnn']:.3f} ocr_hits={r['ocr_hits']} FPR={1-mc['tn']/(mc['tn']+mc['fp']):.3f}")
json.dump(out,open('robust.json','w'),default=float,indent=1)
