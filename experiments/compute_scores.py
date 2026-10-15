"""
compute_scores.py  (revision experiments)
Caches, for every evaluation image, its raw score against EVERY one of the
68 held-out gallery images for each matcher (proposed modules + baselines),
plus the gallery-independent OCR score. Any gallery subset / size k can then
be evaluated exactly by taking the max over the chosen gallery columns.

Usage: python3 experiments/compute_scores.py [--perturb NAME] [--workers 3]
"""
import argparse, csv, glob, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import logo_matcher, color_inspector, typo_detective

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRANDS = ['NIKE', 'PUMA', 'ADIDAS', 'GUCCI', 'LV']
MODEL = os.path.expanduser('~/models/mnv2_feat.onnx')

# ---------------- perturbations for robustness study ----------------
def perturb(img, name):
    if name is None or name == 'none':
        return img
    h, w = img.shape[:2]
    if name == 'blur':          # defocus / motion-ish blur
        return cv2.GaussianBlur(img, (0, 0), 3)
    if name == 'jpeg':          # heavy re-compression (marketplace re-upload)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 20])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if name == 'lowlight':      # under-exposure + sensor noise
        out = img.astype(np.float32) * 0.45
        out += np.random.default_rng(0).normal(0, 8, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    if name == 'lowres':        # small thumbnail upscaled back
        s = cv2.resize(img, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    if name == 'rotate':        # 30 deg handheld tilt
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if name == 'perspective':   # oblique viewpoint
        d = 0.15
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * d, h * d * 0.5], [w * (1 - d), 0], [w, h], [0, h * (1 - d * 0.5)]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if name == 'occlusion':     # 25% area occluded (hand / sticker / watermark)
        out = img.copy()
        oh, ow = int(h * 0.5), int(w * 0.5)
        y, x = int(h * 0.25), int(w * 0.25)
        out[y:y + oh, x:x + ow] = (128, 128, 128)
        return out
    if name == 'colorcast':     # warm white balance (indoor tungsten)
        out = img.astype(np.float32)
        out[..., 2] *= 1.25; out[..., 0] *= 0.8
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(name)

MAX_SIDE = 1600   # long-side cap applied to every image before all modules
def cap(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return img if s >= 1 else cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)

# ---------------- feature extractors ----------------
_orb = None; _sift = None; _hog = None; _sess = None
def init_worker():
    global _orb, _sift, _hog, _sess, G
    cv2.setNumThreads(1)
    _orb = cv2.ORB_create(nfeatures=logo_matcher.ORB_FEATURES)
    _sift = cv2.SIFT_create(nfeatures=1000)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    _sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
    G = build_gallery()

def hs_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [color_inspector.H_BINS, color_inspector.S_BINS], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h

def phash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    return d[1:] > np.median(d[1:])

def hogvec(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
    from skimage.feature import hog as _skhog
    v = _skhog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

def cnnvec(img):
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    v = _sess.run(['464'], {'input': x})[0].flatten()
    return v / (np.linalg.norm(v) + 1e-8)

def feats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = {}
    t0 = time.perf_counter(); okp, odes = _orb.detectAndCompute(gray, None); t['orb_extract'] = time.perf_counter() - t0
    t0 = time.perf_counter(); hh = hs_hist(img); t['color_extract'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if os.environ.get('NO_SIFT') == '1':
        skp, sdes = [], None
    else:
        skp, sdes = _sift.detectAndCompute(gray, None)
    t['sift_extract'] = time.perf_counter() - t0
    t0 = time.perf_counter(); ph = phash(img); t['phash'] = time.perf_counter() - t0
    t0 = time.perf_counter(); hv = hogvec(img); t['hog'] = time.perf_counter() - t0
    t0 = time.perf_counter(); cv = cnnvec(img); t['cnn'] = time.perf_counter() - t0
    return dict(okp=okp, odes=odes, hh=hh, skp=skp, sdes=sdes, ph=ph, hv=hv, cv=cv, n_orb=len(okp)), t

def ratio_matches(d1, d2, norm):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return 0
    bf = cv2.BFMatcher(norm, crossCheck=False)
    good = 0
    for p in bf.knnMatch(d1, d2, k=2):
        if len(p) == 2 and p[0].distance < logo_matcher.LOWE_RATIO * p[1].distance:
            good += 1
    return good

def gallery_paths():
    return sorted(glob.glob(os.path.join(ROOT, 'dataset/gallery/*.jpg')))

def build_gallery():
    out = []
    for p in gallery_paths():
        f, _ = feats(cap(cv2.imread(p, cv2.IMREAD_COLOR)))
        out.append(f)
    return out

PERT = None
def score_one(args):
    path, = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if min(img.shape[:2]) < 16:   # degenerate file (e.g. 4x1 px): every module fails, as in main.py
        k = len(G)
        return dict(path=path, orb=np.zeros(k, np.int32), col=np.full(k, -1, np.float32), sift=np.zeros(k, np.int32),
                    ph=np.full(k, 63, np.int32), hog=np.zeros(k, np.float32), cnn=np.zeros(k, np.float32),
                    ocr=0.0, txt='', n_orb=0, h=img.shape[0], w=img.shape[1],
                    t={k2: 0.0 for k2 in ['orb_extract','color_extract','sift_extract','phash','hog','cnn','orb_match_all','color_match_all','sift_match_all','ocr']})
    img = perturb(cap(img), PERT)
    f, t = feats(img)
    k = len(G)
    orb = np.zeros(k, np.int32); col = np.zeros(k, np.float32); sift = np.zeros(k, np.int32)
    ph = np.zeros(k, np.int32); hog = np.zeros(k, np.float32); cnn = np.zeros(k, np.float32)
    t0 = time.perf_counter()
    for i, g in enumerate(G):   # same argument order as logo_matcher.verify_gallery (gallery = query)
        orb[i] = ratio_matches(g['odes'], f['odes'], cv2.NORM_HAMMING)
    t['orb_match_all'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i, g in enumerate(G):
        col[i] = cv2.compareHist(g['hh'], f['hh'], cv2.HISTCMP_CORREL)
    t['color_match_all'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i, g in enumerate(G):
        sift[i] = ratio_matches(g['sdes'], f['sdes'], cv2.NORM_L2)
    t['sift_match_all'] = time.perf_counter() - t0
    for i, g in enumerate(G):
        ph[i] = int(np.count_nonzero(g['ph'] != f['ph']))
        hog[i] = float(g['hv'] @ f['hv'])
        cnn[i] = float(g['cv'] @ f['cv'])
    # OCR exactly as typo_detective, but on the (possibly perturbed) array
    t0 = time.perf_counter()
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    import pytesseract
    txt = typo_detective._clean_text(pytesseract.image_to_string(th, config=typo_detective.OCR_CONFIG))
    ocr = 0.0
    if txt:
        ocr = max(typo_detective._best_substring_ratio(b, txt) for b in BRANDS)
    t['ocr'] = time.perf_counter() - t0
    h, w = img.shape[:2]
    return dict(path=path, orb=orb, col=col, sift=sift, ph=ph, hog=hog, cnn=cnn, ocr=ocr, txt=txt,
                n_orb=f['n_orb'], h=h, w=w, t=t)

def load_meta():
    rows = []
    for cls in ['genuine', 'counterfeit']:
        man = {}
        with open(os.path.join(ROOT, f'dataset/{cls}/_manifest.csv')) as fh:
            for r in csv.DictReader(fh):
                man[r['output_filename']] = r
        for p in sorted(glob.glob(os.path.join(ROOT, f'dataset/{cls}/*.jpg'))):
            r = man[os.path.basename(p)]
            rows.append(dict(path=p, label=1 if cls == 'genuine' else 0,
                             brand=r['source_seed_image'].split('_')[0].upper(),
                             seed=r['source_seed_image'], aug=r['type'] != 'original'))
    return rows

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--perturb', default='none')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--originals_only', action='store_true')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--no_sift', action='store_true')
    ap.add_argument('--budget', type=float, default=140)
    a = ap.parse_args()
    PERT = a.perturb
    os.environ['NO_SIFT'] = '1' if a.no_sift else '0'
    meta = load_meta()
    if a.originals_only:
        import random
        meta = [m for m in meta if not m['aug']]
        rng = random.Random(7)   # seeded, class-balanced 200/200 robustness sample
        meta = sorted(rng.sample([m for m in meta if m['label'] == 1], 200) +
                      rng.sample([m for m in meta if m['label'] == 0], 200), key=lambda m: m['path'])
    meta = meta[a.start:a.end]
    from multiprocessing import Pool
    import pickle
    cdir = os.path.join(ROOT, f'experiments/results/cache_{a.perturb}')
    os.makedirs(cdir, exist_ok=True)
    todo = [(j, m) for j, m in enumerate(meta) if not os.path.exists(os.path.join(cdir, f'{j:05d}.pkl'))]
    t0 = time.time()
    if todo:
        with Pool(a.workers, initializer=init_worker) as pool:
            for (j, m), r in zip(todo, pool.imap(score_one, [(m['path'],) for _, m in todo])):
                with open(os.path.join(cdir, f'{j:05d}.pkl'), 'wb') as fh:
                    pickle.dump(r, fh)
                if time.time() - t0 > a.budget:
                    print('BUDGET', flush=True); pool.terminate(); break
    left = sum(not os.path.exists(os.path.join(cdir, f'{j:05d}.pkl')) for j in range(len(meta)))
    print('LEFT', left, flush=True)
    if left:
        sys.exit(0)
    res = [pickle.load(open(os.path.join(cdir, f'{j:05d}.pkl'), 'rb')) for j in range(len(meta))]
    assert all(r['path'] == m['path'] for r, m in zip(res, meta))
    gp = gallery_paths()
    out = os.path.join(ROOT, f'experiments/results/scores_{a.perturb}.npz')
    np.savez_compressed(out,
        path=np.array([os.path.relpath(m['path'], ROOT) for m in meta]),
        label=np.array([m['label'] for m in meta]), brand=np.array([m['brand'] for m in meta]),
        seed=np.array([m['seed'] for m in meta]), aug=np.array([m['aug'] for m in meta]),
        gallery=np.array([os.path.relpath(p, ROOT) for p in gp]),
        gallery_brand=np.array([os.path.basename(p).split('_')[0].upper() for p in gp]),
        orb=np.stack([r['orb'] for r in res]), col=np.stack([r['col'] for r in res]),
        sift=np.stack([r['sift'] for r in res]), ph=np.stack([r['ph'] for r in res]),
        hog=np.stack([r['hog'] for r in res]), cnn=np.stack([r['cnn'] for r in res]),
        ocr=np.array([r['ocr'] for r in res]), txt=np.array([r['txt'] for r in res]),
        n_orb=np.array([r['n_orb'] for r in res]), hw=np.array([[r['h'], r['w']] for r in res]),
        times=np.array([[r['t'][k] for k in sorted(r['t'])] for r in res]),
        time_keys=np.array(sorted(res[0]['t'])))
    print('DONE', out, time.time() - t0)
