from common import *
import json
y=D['label']==1; idx=gidx_paperstyle(D,20,42)
o,c,t=mods(D,idx); v=o.astype(int)+c+t; p=v>=2
keep=~(np.load('gt_mask.npy')|np.load('cross_mask.npy'))
print('clean subset n',keep.sum(),met(y[keep],p[keep]))
CV=np.load('../results/color_variants.npz'); bg=CV['none__bg']
FN=y&~p; FP=~y&p; TP=y&p; TN=~y&~p
nk=D['n_orb']; hw=D['hw']; mp=hw[:,0]*hw[:,1]/1e6
res={}
for nm,m in [('TP',TP),('FN',FN),('FP',FP),('TN',TN)]:
    res[nm]=dict(n=int(m.sum()),bg=float(bg[m].mean()),orbpass=float(o[m].mean()),colpass=float(c[m].mean()),txtpass=float(t[m].mean()),kp=float(np.median(nk[m])),mp=float(np.median(mp[m])),ocr_any=float(np.mean([len(s)>0 for s in D['txt'][m]])))
    print(nm,{k:round(x,3) for k,x in res[nm].items()})
# FN breakdown: which modules failed
print('FN pattern (o,c,t):',{str(k):int(((o==k[0])&(c==k[1])&(t==k[2])&FN).sum()) for k in [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]})
print('FP pattern:',{str(k):int(((o==k[0])&(c==k[1])&(t==k[2])&FP).sum()) for k in [(1,1,0),(1,0,1),(0,1,1),(1,1,1)]})
# bg-binned error
bins=[0,0.1,0.3,0.6,1.01]
for a,b in zip(bins[:-1],bins[1:]):
    m=(bg>=a)&(bg<b)
    print(f'bg {a}-{b}: n={m.sum()} gen={y[m].sum()} recall={p[m&y].mean():.3f} FPR={p[m&~y].mean():.3f}')
res['bins']=[(a,b,int(((bg>=a)&(bg<b)).sum()),int(y[(bg>=a)&(bg<b)].sum()),float(p[(bg>=a)&(bg<b)&y].mean()),float(p[(bg>=a)&(bg<b)&~y].mean())) for a,b in zip(bins[:-1],bins[1:])]
# OCR detail
print('OCR nonempty frac',np.mean([len(s)>0 for s in D['txt']]),'OCR>=0.8 genuine',t[y].sum(),'cf',t[~y].sum())
# per brand
for b in sorted(set(D['brand'])):
    m=D['brand']==b; mm=met(y[m],p[m]); print(b,m.sum(),y[m].sum(),{k:round(x,3) for k,x in mm.items() if k in('acc','pr','rc','f1')})
    res['brand_'+b]=dict(n=int(m.sum()),g=int(y[m].sum()),**mm)
# augmented vs original
for a in [False,True]:
    m=D['aug']==a; print('aug',a,m.sum(),{k:round(x,3) for k,x in met(y[m],p[m]).items() if k in('acc','pr','rc','f1')})
# select failure examples: FN with highest bg? pick representative
rng=np.random.default_rng(3)
def pick(m,key,n=2,rev=True):
    ii=np.where(m)[0]; ii=ii[np.argsort(key[ii])]; ii=ii[::-1] if rev else ii; return list(ii[:n])
ex={'FN_lowbg':pick(FN,-bg,2),'FN_highbg':pick(FN&(bg>0.6),D['col'][:,idx].max(1),1,False),'FP_highbg':pick(FP,bg,2),'FP_logo':pick(FP&o,D['orb'][:,idx].max(1),1)}
print({k:[(D['path'][i],round(float(bg[i]),2),int(D['orb'][i,idx].max()),round(float(D['col'][i,idx].max()),2),round(float(D['ocr'][i]),2)) for i in v] for k,v in ex.items()})
json.dump(res,open('fail.json','w'),default=float,indent=1)
json.dump({k:[int(i) for i in v] for k,v in ex.items()},open('ex.json','w'))
