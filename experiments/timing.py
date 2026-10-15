"""Single-process latency of the deployed cascade (gallery k=20, long side
capped at 1600 px), measured per module over 60 evaluation images
(30 genuine, 30 counterfeit, seeded sample). Includes image decode."""
import os, sys, time, random, statistics, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_scores as c
cv2.setNumThreads(1)
c.init_worker()
G = c.G
idx = random.Random(42).sample(range(len(G)), 20)
G20 = [G[i] for i in idx]
meta = c.load_meta()
rng = random.Random(1)
sample = rng.sample([m for m in meta if m['label'] == 1 and not m['aug']], 30) + rng.sample([m for m in meta if m['label'] == 0 and not m['aug']], 30)
import pytesseract
from modules import typo_detective as td
T = {k: [] for k in ['decode+resize', 'logo', 'color', 'text', 'total']}
for m in sample:
    t_all = time.perf_counter()
    t0 = time.perf_counter(); img = c.cap(cv2.imread(m['path'])); T['decode+resize'].append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    kp, des = c._orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    best = max(c.ratio_matches(g['odes'], des, cv2.NORM_HAMMING) for g in G20)
    T['logo'].append(time.perf_counter() - t0)
    t0 = time.perf_counter(); h = c.hs_hist(img); bc = max(cv2.compareHist(g['hh'], h, cv2.HISTCMP_CORREL) for g in G20); T['color'].append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    txt = td._clean_text(pytesseract.image_to_string(th, config=td.OCR_CONFIG))
    T['text'].append(time.perf_counter() - t0)
    T['total'].append(time.perf_counter() - t_all)
for k, v in T.items():
    v = np.array(v) * 1000
    print(f'{k:14s} mean={v.mean():7.1f} ms  median={np.median(v):7.1f}  p95={np.percentile(v,95):7.1f}')
print('mean image MP', np.mean([np.prod(c.cap(cv2.imread(m["path"])).shape[:2]) for m in sample]) / 1e6)
