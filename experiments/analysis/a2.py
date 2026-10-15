from common import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
y=(D['label']==1).astype(int); groups=D['seed']
# group by seed AND label to keep aug copies together
groups=np.array([f'{l}_{s}' for l,s in zip(D['label'],D['seed'])])
def scores(d,idx):
    return {'ORB (ours, logo)':d['orb'][:,idx].max(1).astype(float),
            'HS-color (ours)':d['col'][:,idx].max(1),
            'OCR text (ours)':d['ocr'],
            'SIFT matching':d['sift'][:,idx].max(1).astype(float),
            'pHash NN':-d['ph'][:,idx].min(1).astype(float),
            'HOG NN':d['hog'][:,idx].max(1),
            'MobileNetV2 emb. NN':d['cnn'][:,idx].max(1)}
cv=StratifiedGroupKFold(5,shuffle=True,random_state=0)
folds=list(cv.split(np.zeros(len(y)),y,groups))
def calib_eval(s):
    # threshold picked on train folds (max acc), applied to test fold
    pred=np.zeros(len(y),bool)
    for tr,te in folds:
        u=np.unique(s[tr]); 
        if len(u)>400: u=np.quantile(s[tr],np.linspace(0,1,401))
        best=max(u,key=lambda th:((s[tr]>=th)==y[tr]).mean())
        pred[te]=s[te]>=best
    return met(y,pred)
import json
out={}
for k,seed in [(20,42),(68,42)]:
    idx=gidx_paperstyle(D,k,seed)
    S=scores(D,idx)
    print('== k',k)
    for nm,s in S.items():
        m=calib_eval(s); auc=roc_auc_score(y,s)
        print(f'{nm:24s} AUC={auc:.3f} acc={m["acc"]:.3f} P={m["pr"]:.3f} R={m["rc"]:.3f} F1={m["f1"]:.3f}')
        out[f'{k}|{nm}']=dict(auc=auc,**m)
    o,c,t=mods(D,idx); v=o.astype(int)+c+t
    for nm,p in [('cascade>=2 fixed',v>=2),('cascade>=1 fixed',v>=1),('color fixed',c)]:
        m=met(y,p);print(f'{nm:24s} acc={m["acc"]:.3f} P={m["pr"]:.3f} R={m["rc"]:.3f} F1={m["f1"]:.3f}');out[f'{k}|{nm}']=m
    # LR fusion of our 3 scores (few params)
    X=np.c_[S['ORB (ours, logo)'],S['HS-color (ours)'],S['OCR text (ours)']]
    pred=np.zeros(len(y));
    for tr,te in folds:
        mdl=make_pipeline(StandardScaler(),LogisticRegression()).fit(X[tr],y[tr]); pred[te]=mdl.predict_proba(X[te])[:,1]
    m=met(y,pred>=0.5); print(f'{"weighted fusion (LR,3w)":24s} AUC={roc_auc_score(y,pred):.3f} acc={m["acc"]:.3f} P={m["pr"]:.3f} R={m["rc"]:.3f} F1={m["f1"]:.3f}');out[f'{k}|fusion']=dict(auc=roc_auc_score(y,pred),**m)
# supervised references (need counterfeit labels + training)
E=np.load('../results/dup_features.npz')
emb=E['emb'][:1600]; assert all(a==b for a,b in zip(E['path'][:1600],D['path']))
for nm,X in [('Supervised LR on MobileNetV2 emb.',emb)]:
    pred=np.zeros(len(y))
    for tr,te in folds:
        mdl=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=3000)).fit(X[tr],y[tr]); pred[te]=mdl.predict_proba(X[te])[:,1]
    m=met(y,pred>=0.5); print(f'{nm} AUC={roc_auc_score(y,pred):.3f} acc={m["acc"]:.3f} P={m["pr"]:.3f} R={m["rc"]:.3f} F1={m["f1"]:.3f}');out['sup']=dict(auc=roc_auc_score(y,pred),**m)
json.dump(out,open('baselines.json','w'),default=float,indent=1)
