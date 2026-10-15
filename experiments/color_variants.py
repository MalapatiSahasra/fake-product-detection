"""Tests the background-shortcut hypothesis for the HS-histogram module and a
proposed fix. For every image: fraction of near-white background pixels, and
gallery correlations for (a) the original HS histogram, (b) chromatic-pixel
masked HS histogram (S>=40, V>=40), (c) gray-world white balance + (b)."""
import os, sys, numpy as np, cv2, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_scores as c
from modules import color_inspector as ci

def wb(img):
    f = img.astype(np.float32); m = f.reshape(-1, 3).mean(0) + 1e-6
    return np.clip(f * (m.mean() / m), 0, 255).astype(np.uint8)

def hist(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = None
    if mask:
        m = ((hsv[..., 1] >= 40) & (hsv[..., 2] >= 40)).astype(np.uint8)
        if m.sum() < 50: m = None
    h = cv2.calcHist([hsv], [0, 1], m, [ci.H_BINS, ci.S_BINS], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h

def feats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg = float(((hsv[..., 1] < 30) & (hsv[..., 2] > 200)).mean())
    return bg, hist(img, False), hist(img, True), hist(wb(img), True)

gp = c.gallery_paths()
GF = [feats(c.cap(cv2.imread(p))) for p in gp]
meta = c.load_meta()
rng = random.Random(7)
orig = [m for m in meta if not m['aug']]
sub = sorted(rng.sample([m for m in orig if m['label'] == 1], 200) + rng.sample([m for m in orig if m['label'] == 0], 200), key=lambda m: m['path'])
out = {}
jobs = [('none', meta)] + [(p, sub) for p in ['lowlight', 'colorcast', 'occlusion', 'jpeg', 'blur']]
for pert, ms in jobs:
    BG, A, B, C = [], [], [], []
    for m in ms:
        img = cv2.imread(m['path'])
        if min(img.shape[:2]) < 16:
            BG.append(0); A.append(np.full(68, -1.)); B.append(np.full(68, -1.)); C.append(np.full(68, -1.)); continue
        img = c.perturb(c.cap(img), pert)
        bg, ha, hb, hc = feats(img)
        BG.append(bg)
        A.append([cv2.compareHist(g[1], ha, cv2.HISTCMP_CORREL) for g in GF])
        B.append([cv2.compareHist(g[2], hb, cv2.HISTCMP_CORREL) for g in GF])
        C.append([cv2.compareHist(g[3], hc, cv2.HISTCMP_CORREL) for g in GF])
    out[pert] = dict(path=np.array([os.path.relpath(m['path'], c.ROOT) for m in ms]), label=np.array([m['label'] for m in ms]),
                     bg=np.array(BG), a=np.array(A, np.float32), b=np.array(B, np.float32), c=np.array(C, np.float32))
np.savez_compressed(os.path.join(c.ROOT, 'experiments/results/color_variants.npz'),
                    **{f'{k}__{kk}': vv for k, v in out.items() for kk, vv in v.items()}, gallery_bg=np.array([g[0] for g in GF]))
print('DONE')
