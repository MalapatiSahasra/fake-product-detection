from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
y=D['label']==1; ks=[1,2,5,10,20,40,68]
res={'Cascade ($\\geq$2 of 3)':[],'Color alone':[],'Logo alone':[]}
for k in ks:
    rows=[]
    for s in range(20 if k<68 else 1):
        idx=random.Random(1000+s).sample(range(68),k); o,c,t=mods(D,idx); v=o.astype(int)+c+t
        rows.append([met(y,v>=2)['f1'],met(y,c)['f1'],met(y,o)['f1']])
    a=np.array(rows)
    for j,kk in enumerate(res): res[kk].append((a[:,j].mean(),a[:,j].std()))
plt.rcParams.update({'font.size':8,'font.family':'serif'})
fig,ax=plt.subplots(figsize=(3.4,2.1))
for (nm,v),col,mk in zip(res.items(),['#1f4e79','#c55a11','#7f7f7f'],['o','s','^']):
    m=np.array([a for a,b in v]); s=np.array([b for a,b in v])
    ax.plot(ks,m,marker=mk,ms=3.5,lw=1.3,color=col,label=nm); ax.fill_between(ks,m-s,m+s,color=col,alpha=0.15,lw=0)
ax.set_xscale('log'); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.set_xlabel('Gallery size $k$ (genuine reference images)'); ax.set_ylabel('F1-score')
ax.set_ylim(0,0.75); ax.grid(alpha=0.3,lw=0.5); ax.legend(frameon=False,loc='lower right',fontsize=7)
fig.tight_layout(pad=0.3); fig.savefig('fig_kshot.pdf'); print({k:[(round(a,3),round(b,3)) for a,b in v] for k,v in res.items()})
idx=gidx_paperstyle(D,20,42); import collections; print(collections.Counter(D['gallery_brand'][idx])); print(collections.Counter(D['gallery_brand']))
