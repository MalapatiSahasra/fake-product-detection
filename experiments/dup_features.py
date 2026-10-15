"""Per-image pHash + MobileNetV2 embedding for every evaluation and gallery
image, used to audit near-duplicates (gallery/test leakage and cross-label
duplicates) in the source dataset."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_scores as c
c.init_worker()
meta = c.load_meta()
paths = [m['path'] for m in meta] + c.gallery_paths()
P, E = [], []
for p in paths:
    img = cv2.imread(p)
    if img is None or min(img.shape[:2]) < 16:
        P.append(np.zeros(63, bool)); E.append(np.zeros(1280, np.float32)); continue
    img = c.cap(img)
    P.append(c.phash(img)); E.append(c.cnnvec(img))
np.savez_compressed(os.path.join(c.ROOT, 'experiments/results/dup_features.npz'),
    path=np.array([os.path.relpath(p, c.ROOT) for p in paths]), ph=np.array(P), emb=np.array(E, np.float32))
print('DONE', len(paths))
