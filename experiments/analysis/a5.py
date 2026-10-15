from common import *
import json
E=np.load('../results/dup_features.npz'); P=E['ph']; X=E['emb']; paths=E['path']
n=1600; lab=D['label']; seed=D['seed']; aug=D['aug']
valid=X.any(1)
S=X@X.T
H=(P[:,None,:]!=P[None,:,:]).sum(2)
dup=(H<=4)&(S>=0.95)&valid[:,None]&valid[None,:]
np.fill_diagonal(dup,False)
# gallery-test
gt=dup[:n,n:]
print('test images with near-dup in gallery:',gt.any(1).sum(),' genuine:',gt.any(1)[lab==1].sum(),' cf:',gt.any(1)[lab==0].sum())
# within test, exclude same seed pairs
tt=dup[:n,:n].copy()
same=seed[:,None]==seed[None,:]
same&= lab[:,None]==lab[None,:]
tt&=~same
cross=tt&(lab[:,None]!=lab[None,:])
within=tt&(lab[:,None]==lab[None,:])
print('cross-label dup pairs:',np.triu(cross).sum(),'images involved',cross.any(1).sum())
print('same-label dup pairs (diff seed):',np.triu(within).sum(),'images',within.any(1).sum())
print('exact-ish (H==0,S>.99) cross',np.triu(cross&(H[:n,:n]==0)).sum())
ex=[(paths[i],paths[j]) for i,j in zip(*np.where(np.triu(cross)))][:6];print(ex)
gx=[(paths[i],paths[n+j]) for i,j in zip(*np.where(gt))][:6];print(gx)
idx=gidx_paperstyle(D,20,42)
print('in k=20 gallery dup hits',gt[:,idx].any(1).sum())
json.dump(dict(gal=int(gt.any(1).sum()),gal_g=int(gt.any(1)[lab==1].sum()),gal_c=int(gt.any(1)[lab==0].sum()),cross_pairs=int(np.triu(cross).sum()),cross_imgs=int(cross.any(1).sum()),within_pairs=int(np.triu(within).sum()),within_imgs=int(within.any(1).sum()),k20=int(gt[:,idx].any(1).sum())),open('dup.json','w'))
np.save('gt_mask.npy',gt.any(1)); np.save('cross_mask.npy',cross.any(1))
