#!/usr/bin/env python3
"""
Crater Learn  -  teach the detector YOUR craters (and what is NOT a crater), refine them to sub-pixel,
and match crater PATTERNS between images taken under different light / angle.

Put this file next to crater_lens.py (unchanged):
    pip install numpy opencv-python pillow scipy scikit-learn joblib
    python crater_learn.py                       # GUI
    python crater_learn.py --selftest            # synthetic check (no files needed)
    python crater_learn.py --image a.png --model output/model.joblib --out output          # headless detect
    python crater_learn.py --match a.png b.png --model output/model.joblib --out output    # headless pattern match

GUI workflow
  1 Open image  ->  2 draw with the mouse:  tool "crater" = drag centre->rim (cyan),  tool "NOT crater" = red
  3 Train ML    ->  4 Detect (ML)           ->  5 fix mistakes with tools "accept" / "reject" -> Train again
  6 Match pattern... (second image)  ->  craters become tie points that do not depend on lighting
Everything is stored in the output folder:
  annotations.csv  (your labels - plain CSV)   model.joblib (the trained model)   train_report.json
  <image>/craters.csv + craters.json + overlay.png      match_<A>_vs_<B>/matches.csv + transform.json + overlay.png
"""
import argparse, csv, itertools, json, math, os, sys, time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_lens():
    """Find your Crater Lens detector: crater_lens.py, or any crater_lens*.py (e.g. crater_lens_ML.py) in this folder or the
    one above.  A file counts only if it really contains the detector (run_pipeline + sample), so the matcher is never picked."""
    import importlib.util
    here = Path(__file__).resolve().parent
    try:
        import crater_lens as m
        if hasattr(m, "run_pipeline") and hasattr(m, "sample"): return m
    except ImportError:
        pass
    found = []
    for d in (here, here.parent):
        for f in sorted(d.glob("crater_lens*.py"), key=lambda p: p.stat().st_mtime, reverse=True):
            t = f.read_text(errors="ignore")
            if "def run_pipeline" in t and "def sample" in t and "def register(" not in t: found.append(f)
    if not found:
        raise SystemExit("\nCannot find your Crater Lens detector.\n"
                         f"Copy crater_lens.py into this folder:\n    {here}\n"
                         "(it is the file that contains 'def run_pipeline' - the original detector you uploaded first).\n")
    spec = importlib.util.spec_from_file_location("crater_lens", found[0]); m = importlib.util.module_from_spec(spec)
    sys.modules["crater_lens"] = m; spec.loader.exec_module(m); print(f"[crater_learn] using detector: {found[0].name}"); return m


CL = _load_lens()

A_, R_ = 24, 10                                   # polar sampling: angles x radii
ANG = np.linspace(0, 2 * np.pi, A_, endpoint=False)
COS, SIN = np.cos(ANG), np.sin(ANG)
RAD = np.linspace(0.4, 1.6, R_)                   # in units of the crater radius


# =============================================================== features (rotation + illumination tolerant)
def local_norm(g, sigma=20):
    m = cv2.GaussianBlur(g, (0, 0), sigma)
    v = cv2.GaussianBlur(g * g, (0, 0), sigma) - m * m
    return (g - m) / np.sqrt(np.maximum(v, 4.0))


class Feat:
    """Everything is measured on a locally contrast-normalised copy; rings are described by rotation-invariant
    numbers (FFT magnitudes), so the learned model does not care where the sun is."""
    def __init__(self, gray):
        g = gray.astype(np.float32)
        self.ln = local_norm(g)
        b = cv2.GaussianBlur(self.ln, (0, 0), 1.2)
        self.gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) / 8
        self.gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) / 8
        self.mag = cv2.magnitude(self.gx, self.gy)
        self.g95 = float(np.percentile(self.mag, 95)) + 1e-6
        self.shape = g.shape

    def polar(self, C, rad=RAD, ang=ANG):
        C = np.asarray(C, np.float32).reshape(-1, 3); N = len(C)
        co, si = np.cos(ang), np.sin(ang)
        X = C[:, 0, None, None] + C[:, 2, None, None] * rad[None, :, None] * co[None, None, :]
        Y = C[:, 1, None, None] + C[:, 2, None, None] * rad[None, :, None] * si[None, None, :]
        shp = (N, len(rad), len(ang))
        v = CL.sample(self.ln, X, Y).reshape(shp)
        gx = CL.sample(self.gx, X, Y).reshape(shp); gy = CL.sample(self.gy, X, Y).reshape(shp)
        return v, gx * co + gy * si, np.hypot(gx, gy)

    def __call__(self, C):
        C = np.asarray(C, np.float32).reshape(-1, 3); N = len(C)
        if N == 0: return np.zeros((0, 6 * R_ + 8 * R_ + 7), np.float32)
        v, gr, m = self.polar(C); gr, m = gr / self.g95, m / self.g95
        f = [v.mean(2), v.std(2), gr.mean(2), gr.std(2), m.mean(2), m.std(2)]
        fv = np.abs(np.fft.rfft(v, axis=2))[:, :, 1:5] / A_
        fg = np.abs(np.fft.rfft(gr, axis=2))[:, :, 1:5] / A_
        rim = slice(4, 6)
        cov = (np.abs(gr[:, rim, :]).max(1) > 0.6).mean(1)
        rim_m, out_m, in_m = m[:, rim].mean((1, 2)), m[:, 8:].mean((1, 2)), m[:, :3].mean((1, 2))
        inner, outer = v[:, :4].mean((1, 2)), v[:, 7:].mean((1, 2))
        extra = np.stack([cov, rim_m, rim_m / (out_m + 1e-3), rim_m / (in_m + 1e-3), inner - outer, np.abs(inner - outer),
                          v[:, :4].std((1, 2)) / (v[:, 7:].std((1, 2)) + 1e-3)], 1)   # NO size feature: a model must not care how big a crater is (zoom!)
        return np.hstack([a.reshape(N, -1) for a in f] + [fv.reshape(N, -1), fg.reshape(N, -1), extra]).astype(np.float32)


class FeatPyr:
    """Scale-adaptive features: every crater is measured on the pyramid level where its radius is ~14-28 px, so a 90 px crater
    (zoomed-in image) is described exactly like a 14 px crater (zoomed-out image).  Level 0 = full resolution."""
    def __init__(self, gray, base_r=14.0):
        g = gray.astype(np.float32); self.levels = [Feat(g)]; self.shape = g.shape; self.base_r = base_r
        while min(g.shape) > 200 and len(self.levels) < 6:
            g = cv2.pyrDown(g); self.levels.append(Feat(g))
        self.g95 = self.levels[0].g95
    def level_for(self, r):
        return np.clip(np.floor(np.log2(np.maximum(np.asarray(r, np.float64), 1e-3) / self.base_r)).astype(int), 0, len(self.levels) - 1)
    def __call__(self, C):
        C = np.asarray(C, np.float32).reshape(-1, 3); lv = self.level_for(C[:, 2]); out = None
        for L in np.unique(lv):
            m = lv == L; sc = 2.0 ** L; Cl = C[m].copy()
            Cl[:, 0] = (Cl[:, 0] + .5) / sc - .5; Cl[:, 1] = (Cl[:, 1] + .5) / sc - .5; Cl[:, 2] /= sc
            f = self.levels[L](Cl)
            if out is None: out = np.zeros((len(C), f.shape[1]), np.float32)
            out[m] = f
        return out if out is not None else np.zeros((0, 6 * R_ + 8 * R_ + 7), np.float32)


# =============================================================== candidates
def scan_candidates(F, rmin, rmax, keep=9000, ratio=1.15, stride=0.25):
    """Dense multi-scale scan with a cheap 'is there a ring of radial gradient here?' pre-filter, done on the pyramid level that
    suits each radius.  Finds rims the Hough step gave up on (partly shadowed, low contrast, very big)."""
    out = []; rad3, ang = np.array([0.9, 1.0, 1.1]), np.linspace(0, 2 * np.pi, 24, endpoint=False)
    r = float(rmin)
    while r <= rmax:
        L = int(F.level_for(r)); sc = 2.0 ** L; Fl = F.levels[L]; Hl, Wl = Fl.shape; st = max(2.5, stride * r / sc)
        xs, ys = np.meshgrid(np.arange(0, Wl, st), np.arange(0, Hl, st)); rl = r / sc
        C = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, rl)], 1).astype(np.float32)
        _, gr, _ = Fl.polar(C, rad3, ang); score = (np.abs(gr).max(1).mean(1)) / Fl.g95
        full = np.stack([(C[:, 0] + .5) * sc - .5, (C[:, 1] + .5) * sc - .5, np.full(len(C), r)], 1)
        out.append(np.c_[full, score]); r *= ratio
    a = np.vstack(out); return a[np.argsort(-a[:, 3])][:keep, :3]


def classic_candidates(img, rmin, rmax):
    P = dict(CL.DEFAULTS); P.update(min_score=0.06, min_chain=15, min_r=rmin, max_r=rmax, max_n=60)
    _, res = CL.run_pipeline(img, P)
    return res["cand"].astype(np.float32), res


def same_crater(a, b, k=0.5):
    return math.hypot(a[0] - b[0], a[1] - b[1]) < k * max(a[2], b[2]) and 0.6 < a[2] / b[2] < 1.67


def nms(C, score, k=0.5):
    keep = []
    for i in np.argsort(-score):
        if not any(same_crater(C[i], C[j], k) for j in keep): keep.append(i)
    return keep


# =============================================================== subpixel refinement
def refine_ellipse(F, cx, cy, r0):
    """Slide/stretch/rotate an ellipse so the mean gradient strength under its rim is maximal.
    Positions are real numbers (bilinear sampling) -> sub-pixel centre, axes and angle."""
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False); ct, st = np.cos(t), np.sin(t); la0 = math.log(r0)

    def cost(p):
        x0, y0, la, lb, th = p; a, b = math.exp(la), math.exp(lb); c, s = math.cos(th), math.sin(th)
        x = x0 + a * ct * c - b * st * s; y = y0 + a * ct * s + b * st * c
        m = CL.sample(F.mag, x, y).mean() / F.g95
        pen = 0.3 * ((a - b) / (a + b)) ** 2 + 3.0 * (max(0, abs(la - la0) - 0.3) ** 2 + max(0, abs(lb - la0) - 0.3) ** 2) \
            + 3.0 * max(0, math.hypot(x0 - cx, y0 - cy) / r0 - 0.25) ** 2
        return -m + pen
    p0 = np.array([cx, cy, la0, la0, 0.0]); sim = [p0] + [p0 + d for d in np.diag([0.06 * r0, 0, 0, 0, 0])[:1]]
    sim = np.vstack([p0] + [p0 + np.eye(5)[i] * s for i, s in enumerate([0.06 * r0, 0.06 * r0, 0.06, 0.06, 0.4])])
    res = minimize(cost, p0, method="Nelder-Mead", options=dict(initial_simplex=sim, xatol=0.01, fatol=1e-4, maxiter=250))
    x0, y0, la, lb, th = res.x; a, b = math.exp(la), math.exp(lb)
    if b > a: a, b, th = b, a, th + math.pi / 2
    return dict(cx=float(x0), cy=float(y0), a=float(a), b=float(b), angle=float(math.degrees(th) % 180), r=float(math.sqrt(a * b)),
                rim_strength=float(-res.fun))


# =============================================================== the learner
def jitter(C, rng, n):
    C = np.asarray(C, np.float32).reshape(-1, 3); out = [C]
    for _ in range(n):
        d = rng.normal(0, 0.035, (len(C), 3)).astype(np.float32)
        out.append(np.c_[C[:, 0] + d[:, 0] * C[:, 2], C[:, 1] + d[:, 1] * C[:, 2], C[:, 2] * (1 + d[:, 2])])
    return np.vstack(out)


def relight(gray, rng):
    """Cheap stand-in for 'the sun was somewhere else': random gamma (sun elevation / contrast), a random-direction
    brightness ramp (sun azimuth gradient), contrast change, blur and noise.  Used only to augment TRAINING copies."""
    g = gray.astype(np.float32) / 255.0; H, W = g.shape
    g = np.clip(g, 0, 1) ** rng.uniform(0.6, 1.7)
    ang = rng.uniform(0, 2 * np.pi); yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = g * (1 + rng.uniform(0.15, 0.7) * ((xx * np.cos(ang) + yy * np.sin(ang)) / max(H, W) - 0.5))
    g = (g - 0.5) * rng.uniform(0.6, 1.4) + 0.5
    s_ = rng.uniform(0, 1.2)
    if s_ > 0.3: g = cv2.GaussianBlur(g, (0, 0), s_)
    g = g + rng.randn(H, W).astype(np.float32) * rng.uniform(0, 0.03)
    return (np.clip(g, 0, 1) * 255).astype(np.uint8)


def build_training(items, rng, n_aug=0):
    """items: list of dict(gray, pos(N,3), neg(M,3), complete(bool), cands(K,3)).  Returns X, y, w.
    n_aug>0 also measures every circle on re-lit copies of the image.  OFF by default: in my synthetic test it made detection WORSE
    (20 vs 28 craters found at high sun), so it is only an experiment switch."""
    X, y, w = [], [], []
    for it in items:
        Fs = [FeatPyr(it["gray"])] + [FeatPyr(relight(it["gray"], rng)) for _ in range(n_aug)]; m = len(Fs)
        feats = lambda C: np.vstack([F(C) for F in Fs])
        pos, neg = np.asarray(it["pos"], np.float32).reshape(-1, 3), np.asarray(it["neg"], np.float32).reshape(-1, 3)
        if len(pos): P = jitter(pos, rng, 5); X.append(feats(P)); y += [1] * (len(P) * m); w += [1.0] * (len(P) * m)
        if len(neg): N_ = jitter(neg, rng, 3); X.append(feats(N_)); y += [0] * (len(N_) * m); w += [1.0] * (len(N_) * m)
        cands = np.asarray(it.get("cands", np.zeros((0, 3))), np.float32).reshape(-1, 3)
        free = np.array([not any(same_crater(c, p, 0.7) for p in pos) and not any(same_crater(c, n, 0.7) for n in neg) for c in cands], bool) if len(cands) else np.zeros(0, bool)
        if free.any():
            cf = cands[free]
            if it["complete"]: sel, wt = cf[rng.permutation(len(cf))[:max(200, 4 * len(pos) * 6)]], 1.0     # fully labelled image: all unlabeled = NOT crater
            else: sel, wt = cf[rng.permutation(len(cf))[:max(30, 2 * len(pos) * 6)]], 0.3                    # partly labelled: weak negatives
            X.append(feats(sel)); y += [0] * (len(sel) * m); w += [wt] * (len(sel) * m)
    if not X: return None, None, None
    return np.vstack(X), np.array(y), np.array(w)


def train_model(items, seed=0, n_aug=0):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    rng = np.random.RandomState(seed); X, y, w = build_training(items, rng, n_aug)
    if X is None or y.sum() < 2 or (y == 0).sum() < 3: raise ValueError("Need at least 2 craters drawn (or auto-labelled). 'Not crater' examples are optional - the program samples background itself.")
    clf = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rep = dict(n_samples=int(len(y)), n_positive=int(y.sum()), n_negative=int((y == 0).sum()))
    try:                                           # honest estimate: folds never share the jitter copies of one drawn crater
        k = int(min(5, y.sum() // 6 + 1, (y == 0).sum())); 
        if k >= 2:
            p = cross_val_predict(clf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba", params=dict(sample_weight=w))[:, 1]
            rep.update(cv_auc=round(float(roc_auc_score(y, p)), 3), cv_precision=round(float(precision_score(y, p > .5)), 3), cv_recall=round(float(recall_score(y, p > .5)), 3),
                       cv_note="optimistic: jittered copies of one crater can fall in different folds")
    except Exception as ex: rep["cv_error"] = str(ex)
    clf.fit(X, y, sample_weight=w); return clf, rep


def detect_ml(img_bgr, clf, thr=0.5, rmin=10, rmax=200, cands=None, refine=True, log=print):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); F = FeatPyr(gray); t0 = time.time()
    C = [scan_candidates(F, rmin, rmax)]
    if cands is not None and len(cands): C.append(np.asarray(cands, np.float32)[:, :3])
    C = np.vstack(C); p = clf.predict_proba(F(C))[:, 1]
    ok = p >= thr; C, p = C[ok], p[ok]; keep = nms(C, p); C, p = C[keep], p[keep]
    out = []
    for (cx, cy, r), pp in zip(C, p):
        d = refine_ellipse(F.levels[0], float(cx), float(cy), float(r)) if refine else dict(cx=float(cx), cy=float(cy), a=float(r), b=float(r), angle=0.0, r=float(r), rim_strength=0.0)
        d.update(conf=float(pp), source="ml"); out.append(d)
    # second NMS on refined ellipses (refinement can pull two candidates onto the same rim)
    keep = nms(np.array([[d["cx"], d["cy"], d["r"]] for d in out]), np.array([d["conf"] for d in out])) if out else []
    out = [out[i] for i in keep]; out.sort(key=lambda d: -d["r"])
    log(f"ML detection: {len(out)} craters (conf >= {thr:.2f}) from {len(C)} accepted candidates in {time.time() - t0:.1f}s"); return out



def auto_label(img_bgr, rmin=8, rmax=200, min_strength=0.0):
    """Strong craters from the classic crater_lens pipeline -> list of (cx,cy,r).  Used as free starting labels."""
    cands, res = classic_candidates(img_bgr, rmin, rmax)
    return [(d["cx"], d["cy"], math.sqrt(d["a"] * d["b"])) for d in res["final"]], cands


def quick_train(img_bgr, extra_pos=(), neg=(), rmin=8, rmax=200, log=print):
    """One call: classic detector provides the first positives, background provides negatives -> trained model."""
    pos, cands = auto_label(img_bgr, rmin, rmax); pos = list(pos) + list(extra_pos)
    log(f"auto-labelled {len(pos)} craters from the classic detector")
    return train_model([dict(gray=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), pos=pos, neg=list(neg), complete=False, cands=cands)])


def self_train(img_bgr, clf, rounds=2, conf=0.6, rmin=8, rmax=200, log=print):
    """Model's own confident detections become extra labels, then retrain.  A round is KEPT only if the number of detections stays stable
    (-5% .. +40%): a jump means the model is learning its own mistakes, so the round is rejected."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); cands = classic_candidates(img_bgr, rmin, rmax)[0]
    base = detect_ml(img_bgr, clf, 0.5, rmin, rmax, cands, refine=False, log=lambda s: None)
    for k in range(rounds):
        d = detect_ml(img_bgr, clf, conf, rmin, rmax, cands, refine=False, log=lambda s: None)
        if len(d) < 3: break
        new, _ = train_model([dict(gray=gray, pos=[(c["cx"], c["cy"], c["r"]) for c in d], neg=[], complete=False, cands=cands)])
        n2 = detect_ml(img_bgr, new, 0.5, rmin, rmax, cands, refine=False, log=lambda s: None)
        if 0.95 * len(base) <= len(n2) <= 1.4 * len(base): clf, base = new, n2; log(f"self-training round {k + 1}: kept ({len(n2)} craters at conf 0.5)")
        else: log(f"self-training round {k + 1}: rejected (would change {len(base)} -> {len(n2)} craters = unstable), model unchanged"); break
    return clf


# =============================================================== crater-PATTERN matching (like a star-tracker for craters)
def _triangles(P, R, K):
    idx = np.array(list(itertools.combinations(range(min(K, len(P))), 3)))
    if len(idx) == 0: return idx, np.zeros((0, 5))
    p = P[idx]; d = lambda a, b: np.linalg.norm(p[:, a] - p[:, b], axis=1)
    dij, djk, dik = d(0, 1), d(1, 2), d(0, 2)                                # side opposite to vertex 2, 0, 1
    sides = np.stack([djk, dik, dij], 1)                                     # side opposite to vertex 0,1,2
    order = np.argsort(sides, axis=1); idx = np.take_along_axis(idx, order, 1)   # vertex 0 = opposite the shortest side ... vertex 2 = opposite the longest
    ss = np.sort(sides, axis=1); L = ss[:, 2]
    r = R[idx] / L[:, None]
    desc = np.c_[ss[:, 0] / L, ss[:, 1] / L, r]
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]; area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    good = (ss[:, 0] / L > 0.2) & (area / L ** 2 > 0.05)
    return idx[good], desc[good]


def similarity_from(a, b):
    """least-squares similarity b ~ s R a + t  (Umeyama, no reflection).  returns (s, theta, t)"""
    za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]; ma, mb = za.mean(), zb.mean()
    den = (np.abs(za - ma) ** 2).sum()
    if den < 1e-9: return None
    q = (np.conj(za - ma) * (zb - mb)).sum() / den; t = mb - q * ma
    return abs(q), float(np.angle(q)), np.array([t.real, t.imag])


def apply_sim(T, P):
    s, th, t = T; c, si = math.cos(th), math.sin(th)
    return s * np.stack([c * P[:, 0] - si * P[:, 1], si * P[:, 0] + c * P[:, 1]], 1) + t


def match_patterns(CA, CB, K=45, tol=0.06, max_hyp=4000, pos_tol=0.3):
    """CA, CB: lists of dict(cx,cy,r).  Finds the similarity transform A->B that maps the most craters onto craters.
    Triangle descriptors (side ratios + radius/size ratios) are invariant to shift, rotation and scale."""
    if len(CA) < 4 or len(CB) < 4: return None
    CA = sorted(CA, key=lambda d: -d["r"]); CB = sorted(CB, key=lambda d: -d["r"])
    PA = np.array([[d["cx"], d["cy"]] for d in CA]); RA = np.array([d["r"] for d in CA])
    PB = np.array([[d["cx"], d["cy"]] for d in CB]); RB = np.array([d["r"] for d in CB])
    ia, da = _triangles(PA, RA, K); ib, db = _triangles(PB, RB, K)
    if len(ia) == 0 or len(ib) == 0: return None
    tree = cKDTree(db); hyp = []
    for i, d in enumerate(da):
        for j in tree.query_ball_point(d, tol, p=np.inf): hyp.append((np.abs(d - db[j]).max(), i, j))
    hyp.sort(); treeB = cKDTree(PB); best = None
    for _, i, j in hyp[:max_hyp]:
        T = similarity_from(PA[ia[i]], PB[ib[j]])
        if T is None or not (0.1 < T[0] < 10): continue
        Q = apply_sim(T, PA); dist, nn = treeB.query(Q)
        ok = (dist < np.maximum(3.0, pos_tol * RB[nn])) & (RB[nn] / (T[0] * RA) > 0.7) & (RB[nn] / (T[0] * RA) < 1.43)
        n = int(ok.sum())
        if best is None or n > best[0]: best = (n, T, ok, nn)
    if best is None or best[0] < 4: return None
    n, T, ok, nn = best
    for _ in range(3):                             # refit on all inliers (uses the SUB-PIXEL crater centres), re-collect
        T2 = similarity_from(PA[ok], PB[nn[ok]])
        if T2 is None: break
        T = T2; Q = apply_sim(T, PA); dist, nn = treeB.query(Q)
        ok = (dist < np.maximum(2.0, 0.2 * RB[nn])) & (RB[nn] / (T[0] * RA) > 0.75) & (RB[nn] / (T[0] * RA) < 1.33)
    ia_, ib_ = np.nonzero(ok)[0], nn[ok]
    if len(ia_) < 4: return None
    res = np.linalg.norm(apply_sim(T, PA[ia_]) - PB[ib_], axis=1)
    return dict(scale=float(T[0]), rotation_deg=float(math.degrees(T[1])), tx=float(T[2][0]), ty=float(T[2][1]), n_inliers=int(len(ia_)),
                inlier_ratio=float(len(ia_) / min(len(CA), len(CB))), rmse_px=float(math.sqrt((res ** 2).mean())),
                pairs=[(int(a), int(b), float(r)) for a, b, r in zip(ia_, ib_, res)], CA=CA, CB=CB)


# =============================================================== PATTERNS: one anchor crater + lines to its neighbours
MAX_PATTERNS_PER_IMAGE, MIN_MEMBERS, MAX_MEMBERS = 4, 3, 7


def _c3(c): return [float(c[0]), float(c[1]), float(c[2])]


def sim_from_two(q1, q2, p1, p2):
    """similarity (scale+rotation+shift, NO mirror) that sends q1->p1 and q2->p2"""
    zq, zp = complex(q2[0] - q1[0], q2[1] - q1[1]), complex(p2[0] - p1[0], p2[1] - p1[1]); a = zp / zq
    t = complex(p1[0], p1[1]) - a * complex(q1[0], q1[1])
    return np.array([[a.real, -a.imag, t.real], [a.imag, a.real, t.imag], [0, 0, 1.0]])


def apply_T(T, P):
    P = np.asarray(P, float).reshape(-1, 2); return P @ T[:2, :2].T + T[:2, 2]


def pattern_signature(anchor, members):
    """what makes the pattern unique, independent of zoom / rotation / shift: members' positions with the anchor at the
    origin, nearest member pointing 'up' and its distance = 1, plus sizes in the same unit."""
    a = np.array(anchor[:2], float); M = np.array([m[:2] for m in members], float) - a; d = np.hypot(M[:, 0], M[:, 1]); k = int(np.argmin(d))
    L0, ang0 = d[k], math.atan2(M[k, 1], M[k, 0]); c, s_ = math.cos(-ang0 - math.pi / 2), math.sin(-ang0 - math.pi / 2)
    R = M @ np.array([[c, s_], [-s_, c]]).T / L0
    return dict(anchor_r=round(anchor[2] / L0, 4), members_xy=np.round(R, 4).tolist(), members_r=[round(m[2] / L0, 4) for m in members])


def find_pattern(pat, craters, probe=None, min_frac=0.6, pos_tol=0.22, probe_p=0.25, frame=None, exclude=None, max_cand=2500):
    """Look for a saved pattern among craters (cx,cy,r) of ANY image: unknown zoom, rotation, shift.
    1. every pair of pattern craters x every pair of image craters whose SIZES and DISTANCES agree proposes a similarity;
    2. the proposal is scored by how many pattern craters land on a real crater (position AND size);
    3. pattern craters that were NOT detected (shallow / washed out by the sun) are checked where the pattern says they must be:
       `probe(circles)` returns the crater-probability of the learned model there  ->  'pattern-guided' detection.
    Returns None or dict(found, T, scale, rotation_deg, n, n_det, n_probed, score, members=[...])."""
    Q = np.array([pat["anchor"]] + list(pat["members"]), float); n = len(Q)
    P = np.array(craters, float).reshape(-1, 3)
    if exclude is not None and len(P):
        P = P[np.array([not any(same_crater(p, e, 0.7) for e in exclude) for p in P])]
    if len(P) < 3 or n < 3: return None
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]); lr = np.log(P[:, 2]); tree = cKDTree(P[:, :2]); cl = []
    for i in range(n):
        for j in range(i + 1, n):
            Lq = math.hypot(*(Q[j, :2] - Q[i, :2]))
            if Lq < 1e-6: continue
            la, lb = (lr - math.log(Q[i, 2]))[:, None], (lr - math.log(Q[j, 2]))[None, :]; ld = np.log(D / Lq + 1e-12)
            ok = (np.abs(la - lb) < 0.45) & (np.abs(ld - la) < 0.45) & (np.abs(ld - lb) < 0.45) & (D > 1e-6)
            aa, bb = np.nonzero(ok)
            if len(aa) > max_cand:
                keep = np.argsort(np.abs(ld[aa, bb] - la[aa, 0]))[:max_cand]; aa, bb = aa[keep], bb[keep]
            for a, b in zip(aa, bb):
                T = sim_from_two(Q[i], Q[j], P[a], P[b]); sc = math.hypot(T[0, 0], T[1, 0]); Qt = apply_T(T, Q[:, :2]); d, nn = tree.query(Qt)
                rr = P[nn, 2] / (Q[:, 2] * sc); hit = (d < np.maximum(pos_tol * Q[:, 2] * sc, 2.5)) & (rr > 0.65) & (rr < 1.55); k = int(hit.sum())
                if k >= 3: cl.append(((k, -float(d[hit].mean())), T, sc, hit, nn, Qt))
    if not cl: return None
    cl.sort(key=lambda t: t[0], reverse=True); top, seen = [], set()
    for key, T, sc, hit, nn, Qt in cl:                    # distinct hypotheses only (zoom / rotation / place)
        sg = (round(math.log(sc) / 0.06), round(math.degrees(math.atan2(T[1, 0], T[0, 0])) / 4), round(T[0, 2] / (0.5 * Q[:, 2].max() * sc + 1)), round(T[1, 2] / (0.5 * Q[:, 2].max() * sc + 1)))
        if sg in seen: continue
        seen.add(sg); top.append((T, sc, hit, nn, Qt))
        if len(top) >= (12 if probe is not None else 1): break
    best = None
    for T, sc, hit, nn, Qt in top:
        for _ in range(3):                  # a hypothesis came from only TWO craters: refit on every matched crater, re-check
            if hit.sum() < 3: break
            s_, th_, t_ = similarity_from(Q[hit, :2], P[nn[hit], :2]); co, si = math.cos(th_), math.sin(th_)
            T2 = np.array([[s_ * co, -s_ * si, t_[0]], [s_ * si, s_ * co, t_[1]], [0, 0, 1.0]]); Qt2 = apply_T(T2, Q[:, :2]); d2, nn2 = tree.query(Qt2)
            rr2 = P[nn2, 2] / (Q[:, 2] * s_); hit2 = (d2 < np.maximum(pos_tol * Q[:, 2] * s_, 2.5)) & (rr2 > 0.65) & (rr2 < 1.55)
            if hit2.sum() >= hit.sum(): T, sc, hit, nn, Qt = T2, s_, hit2, nn2, Qt2
            else: break
        probed, pp = np.zeros(n, bool), np.zeros(n)
        if probe is not None and (~hit).any():             # the learned crater model votes on the pattern craters we did not detect
            idx = np.nonzero(~hit)[0]; circ = np.c_[Qt[idx], Q[idx, 2] * sc]
            ins = np.ones(len(idx), bool) if frame is None else (circ[:, 0] > 0) & (circ[:, 0] < frame[1]) & (circ[:, 1] > 0) & (circ[:, 1] < frame[0]) & (circ[:, 2] > 3)
            if ins.any(): p = probe(circ[ins]); pp[idx[ins]] = p; probed[idx[ins]] = p >= probe_p
        dm = float(np.hypot(*(P[nn[hit], :2] - apply_T(T, Q[hit, :2])).T).mean() / (Q[:, 2].mean() * sc)) if hit.any() else 1.0
        rs = Q[:, 2] * sc; dd = np.hypot(*(P[nn, :2] - Qt).T) / rs; lq = np.abs(np.log(P[nn, 2] / rs))
        w = np.where(hit, np.where((dd < 0.10) & (lq < 0.15), 1.0, 0.6), 0.0)                       # tight agreement of position AND size counts fully
        val = w.sum() + 0.4 * float((np.clip(pp - 0.25, 0, 1) * probed).sum()) - 0.5 * dm          # the model's vote is only a tie-breaker
        if best is None or val > best[0]: best = (val, T, sc, hit.copy(), nn.copy(), Qt.copy(), probed.copy(), pp.copy())
    _, T, sc, hit, nn, Qt, probed, pp = best
    matched = hit | probed
    found = bool(matched.sum() >= max(3, math.ceil(min_frac * n)) and hit.sum() >= max(2, math.ceil(0.4 * n)))
    mem = [dict(cx=float(P[nn[k], 0] if hit[k] else Qt[k, 0]), cy=float(P[nn[k], 1] if hit[k] else Qt[k, 1]), r=float(P[nn[k], 2] if hit[k] else Q[k, 2] * sc),
                det=bool(hit[k]), probed=bool(probed[k]), p=float(pp[k])) for k in range(n)]
    return dict(found=found, T=T, scale=float(sc), rotation_deg=float(math.degrees(math.atan2(T[1, 0], T[0, 0]))), n=n, n_det=int(hit.sum()),
                n_probed=int(probed.sum()), score=float(matched.sum() / n), members=mem)


def propose_patterns(craters, max_n=4, k=5):
    """Automatic pattern recognition: pick up to max_n anchors (big craters with neighbours all around), spread over the image,
    each with its k nearest neighbours.  Marks patterns that look like another place in the same image ('ambiguous')."""
    C = np.array(craters, float).reshape(-1, 3); n = len(C)
    if n < MIN_MEMBERS + 1: return []
    rank = np.argsort(np.argsort(C[:, 2])) / max(n - 1, 1); cand = []
    for i in range(n):
        d = np.hypot(*(C[:, :2] - C[i, :2]).T); order = [j for j in np.argsort(d) if j != i and C[j, 2] >= 0.15 * C[i, 2]][:k]
        if len(order) < MIN_MEMBERS: continue
        an = np.sort(np.arctan2(C[order, 1] - C[i, 1], C[order, 0] - C[i, 0])); cov = 1 - np.diff(np.r_[an, an[0] + 2 * np.pi]).max() / (2 * np.pi)
        cand.append((0.45 * cov + 0.25 * min(1, len(order) / k) + 0.30 * rank[i], i, order, float(d[order].max())))
    cand.sort(key=lambda t: -t[0]); chosen = []
    for q, i, order, ext in cand:
        mem = {i, *order}
        if any(len(mem & {c[1], *c[2]}) > 1 or math.hypot(*(C[i, :2] - C[c[1], :2])) < 0.6 * max(ext, c[3]) for c in chosen): continue
        chosen.append((q, i, order, ext))
        if len(chosen) >= max_n: break
    out = []
    for q, i, order, ext in chosen:
        p = dict(anchor=_c3(C[i]), members=[_c3(C[j]) for j in order], quality=round(float(q), 3))
        rep = find_pattern(p, C.tolist(), min_frac=0.8, exclude=[_c3(C[j]) for j in [i, *order]]); p["ambiguous"] = bool(rep and rep["found"]); out.append(p)
    return out


def draw_patterns_overlay(img, pats, found=()):
    v = img.copy()
    for k, p in enumerate(pats):
        if p.get("anchor") is None: continue
        col = [(255, 200, 0), (0, 200, 255), (0, 255, 120), (255, 120, 255)][k % 4]; a = p["anchor"]
        for i, m in enumerate(p["members"]):
            cv2.line(v, (int(a[0]), int(a[1])), (int(m[0]), int(m[1])), col, 2, cv2.LINE_AA); cv2.circle(v, (int(m[0]), int(m[1])), max(2, int(m[2])), col, 2, cv2.LINE_AA)
            cv2.putText(v, str(i + 1), (int(m[0]) - 4, int(m[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(v, (int(a[0]), int(a[1])), max(3, int(a[2])), (0, 0, 255), 3, cv2.LINE_AA); cv2.putText(v, p.get("name", f"P{k + 1}"), (int(a[0]), int(a[1] - a[2] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    for p, r in found:
        ms = r["members"]; a = ms[0]
        for m in ms[1:]:
            cv2.line(v, (int(a["cx"]), int(a["cy"])), (int(m["cx"]), int(m["cy"])), (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(v, (int(m["cx"]), int(m["cy"])), max(2, int(m["r"])), (255, 0, 255) if m["det"] else (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(v, (int(a["cx"]), int(a["cy"])), max(3, int(a["r"])), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(v, f"{p.get('name', '?')}  x{r['scale']:.2f}  {r['rotation_deg']:.0f}deg", (int(a["cx"]), int(a["cy"] - a["r"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    return v


# =============================================================== storage
class Store:
    """output folder:  annotations.csv (labels), model.joblib, train_report.json, per-image results."""
    FIELDS = ["image", "cx", "cy", "r", "label", "source", "scale"]
    def __init__(self, out): self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True); self.ann = self.out / "annotations.csv"; self.proj = self.out / "project.json"
    def load_ann(self):
        if not self.ann.exists(): return []
        with open(self.ann, newline="") as f: return [dict(r, cx=float(r["cx"]), cy=float(r["cy"]), r=float(r["r"]), label=int(r["label"]), scale=float(r["scale"])) for r in csv.DictReader(f)]
    def save_ann(self, rows):
        with open(self.ann, "w", newline="") as f:
            w = csv.DictWriter(f, self.FIELDS); w.writeheader()
            for r in rows: w.writerow({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in self.FIELDS})
    def load_proj(self): return json.loads(self.proj.read_text()) if self.proj.exists() else {"complete": {}}
    def save_proj(self, p): self.proj.write_text(json.dumps(p, indent=1))
    def load_patterns(self):
        p = self.out / "patterns.json"; return json.loads(p.read_text()) if p.exists() else []
    def save_patterns(self, pats):
        keep = [{k: v for k, v in p.items() if k != "status"} for p in pats]
        for p in keep:
            if p.get("anchor") and len(p["members"]) >= 1: p["signature"] = pattern_signature(p["anchor"], p["members"])
        (self.out / "patterns.json").write_text(json.dumps(keep, indent=1))
    def save_model(self, clf, rep):
        import joblib; joblib.dump(clf, self.out / "model.joblib"); (self.out / "train_report.json").write_text(json.dumps(rep, indent=1))
    def load_model(self):
        import joblib; p = self.out / "model.joblib"; return joblib.load(p) if p.exists() else None
    def save_craters(self, name, dets, img, scale, shape):
        d = self.out / Path(name).stem; d.mkdir(exist_ok=True); H, W = shape[:2]
        with open(d / "craters.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "cx", "cy", "a", "b", "angle_deg", "r", "cx_orig", "cy_orig", "r_orig", "confidence", "source", "rim_strength"])
            for i, c in enumerate(dets):
                w.writerow([i, f"{c['cx']:.3f}", f"{c['cy']:.3f}", f"{c['a']:.3f}", f"{c['b']:.3f}", f"{c['angle']:.2f}", f"{c['r']:.3f}",
                            f"{(c['cx'] + .5) / scale - .5:.3f}", f"{(c['cy'] + .5) / scale - .5:.3f}", f"{c['r'] / scale:.3f}", f"{c.get('conf', 1.0):.3f}", c.get("source", ""), f"{c.get('rim_strength', 0):.3f}"])
        (d / "craters.json").write_text(json.dumps({"image": str(name), "w": W, "h": H, "scale_to_original": scale,
            "circles": [[round(c["cx"], 2), round(c["cy"], 2), round(c["r"], 2)] for c in dets], "ellipses": dets}, indent=1))   # same layout as crater_lens detected.json
        cv2.imwrite(str(d / "overlay.png"), draw_overlay(img, dets)); return d


def draw_overlay(img, dets, ids=False):
    v = img.copy()
    for i, c in enumerate(dets):
        col = (0, 255, 0) if c.get("source") == "ml" else (255, 220, 0)
        CL.draw_ellipse(v, c["cx"], c["cy"], c["a"], c["b"], c["angle"], col, 2)
        if ids: cv2.putText(v, str(i), (int(c["cx"]), int(c["cy"])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return v


def save_match(store, nameA, nameB, imgA, imgB, m):
    d = store.out / f"match_{Path(nameA).stem}_vs_{Path(nameB).stem}"; d.mkdir(exist_ok=True)
    with open(d / "matches.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["crater_A", "crater_B", "xA", "yA", "rA", "xB", "yB", "rB", "residual_px"])
        for a, b, r in m["pairs"]:
            A, B = m["CA"][a], m["CB"][b]; w.writerow([a, b, f"{A['cx']:.3f}", f"{A['cy']:.3f}", f"{A['r']:.3f}", f"{B['cx']:.3f}", f"{B['cy']:.3f}", f"{B['r']:.3f}", f"{r:.3f}"])
    (d / "transform.json").write_text(json.dumps({k: v for k, v in m.items() if k not in ("pairs", "CA", "CB")}, indent=1))
    h = max(imgA.shape[0], imgB.shape[0]); canvas = np.zeros((h, imgA.shape[1] + imgB.shape[1], 3), np.uint8)
    canvas[:imgA.shape[0], :imgA.shape[1]] = draw_overlay(imgA, m["CA"]); canvas[:imgB.shape[0], imgA.shape[1]:] = draw_overlay(imgB, m["CB"])
    for a, b, r in m["pairs"]:
        A, B = m["CA"][a], m["CB"][b]
        cv2.line(canvas, (int(A["cx"]), int(A["cy"])), (int(B["cx"] + imgA.shape[1]), int(B["cy"])), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(d / "overlay.png"), canvas); return d, canvas


def load_img(path, max_side):
    CL.MAX_SIDE = max_side; raw = cv2.imread(path)
    if raw is None: raise ValueError(f"cannot read {path}")
    img = CL.load_image(path); return img, img.shape[1] / raw.shape[1]


# =============================================================== GUI
def run_gui(args):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    root = tk.Tk(); root.title("Crater Learn"); root.geometry("1400x900")
    S = dict(img=None, path="", scale=1.0, ann=[], dets=[], store=Store(args.out), clf=None, zoom=1.0, ox=0.0, oy=0.0, photo=None, cands=None, drag=None, pan=None, patterns=[], props=[], cur=None, found=[])
    S["patterns"] = S["store"].load_patterns()
    pstat = tk.StringVar(value="Patterns: none yet - detect craters, then press 'Propose patterns'")
    S["clf"] = S["store"].load_model(); proj = S["store"].load_proj()
    tool = tk.StringVar(value="crater"); thr = tk.DoubleVar(value=0.5); complete = tk.BooleanVar(value=False); show_dets = tk.BooleanVar(value=True)
    bar = ttk.Frame(root, padding=3); bar.pack(fill="x"); bar2 = ttk.Frame(root, padding=3); bar2.pack(fill="x"); bar3 = ttk.Frame(root, padding=3); bar3.pack(fill="x"); bar4 = ttk.Frame(root, padding=(6, 0)); bar4.pack(fill="x")
    cv = tk.Canvas(root, bg="#111", highlightthickness=0, cursor="crosshair"); cv.pack(fill="both", expand=True)
    log = tk.Text(root, height=7, bg="#1b1b1b", fg="#9fe", font=("Menlo", 10)); log.pack(fill="x")

    def say(s): log.insert("end", s + "\n"); log.see("end"); root.update_idletasks()
    def i2c(x, y): return x * S["zoom"] + S["ox"], y * S["zoom"] + S["oy"]
    def c2i(x, y): return (x - S["ox"]) / S["zoom"], (y - S["oy"]) / S["zoom"]
    def mine(): return [a for a in S["ann"] if a["image"] == S["path"]]

    def poly(c):
        t = np.linspace(0, 2 * np.pi, 48); th = math.radians(c["angle"]); a, b = c["a"], c["b"]
        x = c["cx"] + a * np.cos(t) * math.cos(th) - b * np.sin(t) * math.sin(th); y = c["cy"] + a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th)
        return [v for p in zip(*i2c(x, y)) for v in p]

    def redraw(_=None):
        cv.delete("all")
        if S["img"] is None: return
        H, W = S["img"].shape[:2]; cw, ch = max(cv.winfo_width(), 50), max(cv.winfo_height(), 50)
        x0, y0 = c2i(0, 0); x1, y1 = c2i(cw, ch); x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1) + 1), min(H, int(y1) + 1)
        if x1 > x0 and y1 > y0:
            crop = S["img"][y0:y1, x0:x1]; sz = (max(1, int((x1 - x0) * S["zoom"])), max(1, int((y1 - y0) * S["zoom"])))
            disp = cv2.resize(crop, sz, interpolation=cv2.INTER_NEAREST if S["zoom"] > 1 else cv2.INTER_AREA)
            S["photo"] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))); cx, cy = i2c(x0, y0); cv.create_image(cx, cy, anchor="nw", image=S["photo"])
        if show_dets.get():
            for d in S["dets"]:
                if d["conf"] >= thr.get(): cv.create_polygon(poly(d), outline="#3f3" if d["conf"] > .75 else "#fd3", fill="", width=2)
        for a in mine():
            x, y = i2c(a["cx"], a["cy"]); r = a["r"] * S["zoom"]; col = "#0ff" if a["label"] == 1 else "#f44"
            cv.create_oval(x - r, y - r, x + r, y + r, outline=col, width=2, dash=(4, 2) if a["label"] == 0 else ())
        if S["drag"]:
            (xa, ya), (xb, yb) = S["drag"]; r = math.hypot(xb - xa, yb - ya); cv.create_oval(xa - r, ya - r, xa + r, ya + r, outline="#fff", width=1)
        draw_pats()

    def open_image(p=None):
        p = p or filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
        if not p: return
        S["img"], S["scale"] = load_img(p, args.max_side); S["path"] = os.path.abspath(p); S["dets"] = []; S["cands"] = None; S["props"] = []; S["found"] = []; S["cur"] = (my_pats() or [None])[0]
        H, W = S["img"].shape[:2]; cw, ch = max(cv.winfo_width(), 50), max(cv.winfo_height(), 50); S["zoom"] = min(cw / W, ch / H); S["ox"] = S["oy"] = 0
        complete.set(bool(proj["complete"].get(S["path"], False))); say(f"Opened {p}  ({W}x{H}, scale to original = {S['scale']:.3f})  | your labels here: {len(mine())}  | patterns here: {len(my_pats())}"); pstatus(); redraw()

    def classic():
        if S["img"] is None: return
        say("Classic pipeline (crater_lens) + low-threshold candidates ..."); cands, res = classic_candidates(S["img"], args.min_r, args.max_r); S["cands"] = cands
        S["classic"] = res["final"]; say(f"  classic detector found {len(res['final'])} craters; {len(cands)} candidates kept for the ML step")
        S["dets"] = [dict(d, conf=0.99, source="classic", rim_strength=0.0, angle=d["angle"]) for d in res["final"]]; redraw()

    def sync():
        S["store"].save_ann(S["ann"]); proj["complete"][S["path"]] = bool(complete.get()); S["store"].save_proj(proj)

    def train():
        sync(); paths = sorted({a["image"] for a in S["ann"]})
        if not paths: messagebox.showwarning("No labels", "Draw some craters (tool 'crater') and some NOT-craters first."); return
        items = []
        for p in paths:
            img, sc = load_img(p, args.max_side) if p != S["path"] else (S["img"], S["scale"]); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); rows = [a for a in S["ann"] if a["image"] == p]
            if p == S["path"] and S["cands"] is None: classic()
            cands = S["cands"] if p == S["path"] else classic_candidates(img, args.min_r, args.max_r)[0]
            items.append(dict(gray=g, pos=[(a["cx"], a["cy"], a["r"]) for a in rows if a["label"] == 1], neg=[(a["cx"], a["cy"], a["r"]) for a in rows if a["label"] == 0],
                              complete=bool(proj["complete"].get(p, False)), cands=cands))
        say(f"Training on {len(paths)} image(s) ..."); 
        try: S["clf"], rep = train_model(items)
        except Exception as ex: messagebox.showerror("Training failed", str(ex)); return
        S["store"].save_model(S["clf"], rep); say("Trained + saved model.joblib:  " + json.dumps(rep))

    def detect():
        if S["img"] is None: return
        if S["clf"] is None: messagebox.showwarning("No model", "Train the model first (or load one)."); return
        if S["cands"] is None: classic()
        S["dets"] = detect_ml(S["img"], S["clf"], 0.3, args.min_r, args.max_r, S["cands"], log=say); redraw()

    def autolabel():
        if S["img"] is None: return
        pos, S["cands"] = auto_label(S["img"], args.min_r, args.max_r)
        for cx, cy, r in pos: add_ann(cx, cy, r, 1, "auto")
        say(f"Auto-labelled {len(pos)} craters (cyan). Erase wrong ones, add missed ones with 'crater', mark false ones with 'NOT crater', then Train ML."); redraw()

    def selftrain():
        if S["img"] is None or S["clf"] is None: messagebox.showwarning("Need", "Train a model first."); return
        say("Self-training (model's confident detections become labels) ..."); S["clf"] = self_train(S["img"], S["clf"], 2, 0.6, args.min_r, args.max_r, say)
        S["store"].save_model(S["clf"], {"note": "self-trained"}); say("Model updated + saved."); detect()

    def add_ann(cx, cy, r, label, src="manual"):
        S["ann"].append(dict(image=S["path"], cx=float(cx), cy=float(cy), r=float(r), label=label, source=src, scale=S["scale"])); sync()

    def press(e):
        if S["img"] is None: return
        if e.state & 0x1: S["pan"] = (e.x, e.y, S["ox"], S["oy"]); return                       # Shift+drag = pan
        t = tool.get(); x, y = c2i(e.x, e.y)
        if t in ("crater", "NOT crater", "pat member"): S["drag"] = ((e.x, e.y), (e.x, e.y))
        elif t == "pat anchor": pat_click(x, y, "anchor")
        else:
            if t in ("accept", "reject"):
                best = None
                for d in S["dets"]:
                    dist = math.hypot(x - d["cx"], y - d["cy"]) / d["r"]
                    if dist < 1.1 and (best is None or d["r"] < best["r"]): best = d
                if best: add_ann(best["cx"], best["cy"], best["r"], 1 if t == "accept" else 0, "accepted" if t == "accept" else "rejected"); say(f"{t}ed detection r={best['r']:.1f}"); redraw()
            elif t == "erase":
                m = [(math.hypot(x - a["cx"], y - a["cy"]), a) for a in mine()]
                if m:
                    dd, a = min(m, key=lambda z: z[0])
                    if dd < a["r"] * 1.2: S["ann"].remove(a); sync(); redraw()

    def move(e):
        if S["pan"]: S["ox"] = S["pan"][2] + e.x - S["pan"][0]; S["oy"] = S["pan"][3] + e.y - S["pan"][1]; redraw()
        elif S["drag"]: S["drag"] = (S["drag"][0], (e.x, e.y)); redraw()

    def release(e):
        S["pan"] = None
        if S["drag"]:
            (xa, ya), (xb, yb) = S["drag"]; S["drag"] = None; r = math.hypot(xb - xa, yb - ya) / S["zoom"]
            if tool.get() == "pat member":
                if r >= 3: cx, cy = c2i(xa, ya); add_ann(cx, cy, r, 1); pat_click(cx, cy, "member")       # drawn crater becomes a member at once
                else: x, y = c2i(xa, ya); pat_click(x, y, "member")
            elif r >= 3: cx, cy = c2i(xa, ya); add_ann(cx, cy, r, 1 if tool.get() == "crater" else 0)
            redraw()

    def wheel(e):
        f = 1.25 if (getattr(e, "delta", 0) > 0 or e.num == 4) else 0.8; x, y = c2i(e.x, e.y); S["zoom"] *= f; S["ox"] = e.x - x * S["zoom"]; S["oy"] = e.y - y * S["zoom"]; redraw()
    cv.bind("<ButtonPress-1>", press); cv.bind("<B1-Motion>", move); cv.bind("<ButtonRelease-1>", release); cv.bind("<MouseWheel>", wheel); cv.bind("<Button-4>", wheel); cv.bind("<Button-5>", wheel); cv.bind("<Configure>", redraw)

    # ================================================================ PATTERNS
    import uuid
    def pool(low=False):
        """every crater known in this image as (cx,cy,r): yours + detections"""
        out = [(a["cx"], a["cy"], a["r"]) for a in mine() if a["label"] == 1]; neg = [(a["cx"], a["cy"], a["r"]) for a in mine() if a["label"] == 0]
        for d in S["dets"]:
            c = (d["cx"], d["cy"], d["r"])
            if d.get("conf", 1) >= (0.25 if low else thr.get()) and not any(same_crater(c, n) for n in neg) and not any(same_crater(c, o) for o in out): out.append(c)
        return out

    def pick(x, y):
        best = None
        for c in pool():
            if math.hypot(x - c[0], y - c[1]) / c[2] < 1.1 and (best is None or c[2] < best[2]): best = c
        return best

    def my_pats(): return [p for p in S["patterns"] if p["image"] == S["path"]]
    def view_list(): return my_pats() + S["props"]
    def is_saved(p): return any(p is q for q in S["patterns"])

    def pstatus():
        p = S["cur"]; v = view_list(); n = len(my_pats())
        head = f"[{n}/{MAX_PATTERNS_PER_IMAGE} saved for this image]  "
        if p is None: pstat.set(head + ("no pattern selected - 'Propose patterns' or 'New (draw)'" if n == 0 else "use < > to view a pattern")); return
        k = next((i for i, q in enumerate(v) if q is p), 0) + 1
        warn = "   !! this pattern also fits another place in the image (not unique)" if p.get("ambiguous") else ""
        a = p.get("anchor"); pstat.set(head + f"viewing {k}/{len(v)}  {'SAVED' if is_saved(p) else 'PROPOSAL (not saved)'}  anchor r={a[2]:.0f}  neighbours={len(p['members'])}{warn}" if a else head + "new pattern: pick the anchor crater")

    def draw_pat(p, col, width, tag):
        if p.get("anchor") is None: return
        a = p["anchor"]; ax_, ay_ = i2c(a[0], a[1]); ra = a[2] * S["zoom"]; zz = S["zoom"]
        for k, m in enumerate(p["members"]):
            mx, my = i2c(m[0], m[1]); rm = m[2] * zz
            cv.create_line(ax_, ay_, mx, my, fill=col, width=width); cv.create_oval(mx - rm, my - rm, mx + rm, my + rm, outline=col, width=width)
            cv.create_text(mx, my, text=str(k + 1), fill="#fff", font=("Arial", 11, "bold"))
        cv.create_oval(ax_ - ra, ay_ - ra, ax_ + ra, ay_ + ra, outline="#f33", width=width + 1); cv.create_text(ax_, ay_ - ra - 12, text=tag, fill=col, font=("Arial", 11, "bold"))

    def draw_pats():
        for p in my_pats():
            if p is not S["cur"]: draw_pat(p, "#6cf", 2, p.get("name", ""))
        for p in S["props"]:
            if p is not S["cur"]: continue
        if S["cur"] is not None: draw_pat(S["cur"], "#ff0", 4, ("SAVED " if is_saved(S["cur"]) else "PROPOSAL ") + S["cur"].get("name", ""))
        for p, r in S["found"]:
            ms = r["members"]; a = ms[0]; ax_, ay_ = i2c(a["cx"], a["cy"])
            for m in ms[1:]:
                mx, my = i2c(m["cx"], m["cy"]); rm = m["r"] * S["zoom"]; cv.create_line(ax_, ay_, mx, my, fill="#f0f", width=2)
                cv.create_oval(mx - rm, my - rm, mx + rm, my + rm, outline="#f0f" if m["det"] else "#fa0", width=2, dash=() if m["det"] else (4, 3))
            ra = a["r"] * S["zoom"]; cv.create_oval(ax_ - ra, ay_ - ra, ax_ + ra, ay_ + ra, outline="#f33", width=3)
            cv.create_text(ax_, ay_ - ra - 12, text=f"{p.get('name', '?')}  x{r['scale']:.2f}  {r['rotation_deg']:.0f} deg", fill="#f0f", font=("Arial", 11, "bold"))

    def propose():
        if S["img"] is None: return
        room = MAX_PATTERNS_PER_IMAGE - len(my_pats())
        if room <= 0: messagebox.showinfo("Patterns", f"This image already has {MAX_PATTERNS_PER_IMAGE} patterns (the maximum). Delete one first."); return
        c = pool()
        if len(c) < 4: messagebox.showwarning("Patterns", "Need at least 4 craters in this image.\nPress 'Detect (ML)' (or draw craters) first."); return
        props = propose_patterns(c, room, 5)
        if not props: messagebox.showinfo("Patterns", "Could not form a pattern (craters too few or too spread out)."); return
        for i, p in enumerate(props): p.update(id=uuid.uuid4().hex[:8], name=f"proposal {i + 1}", image=S["path"], scale=S["scale"])
        S["props"] = props; S["cur"] = props[0]; say(f"{len(props)} pattern(s) proposed. Look at the yellow pattern: OK = keep, Edit = change it, Delete = reject, > = next proposal."); pstatus(); redraw()

    def step(d):
        v = view_list()
        if not v: return
        i = next((k for k, q in enumerate(v) if q is S["cur"]), -1); S["cur"] = v[(i + d) % len(v)]; pstatus(); redraw()

    def ok_pat():
        p = S["cur"]
        if p is None: return
        if is_saved(p): say("This pattern is already saved."); return
        if p.get("anchor") is None or len(p["members"]) < MIN_MEMBERS: messagebox.showwarning("Pattern", f"A pattern needs an anchor and at least {MIN_MEMBERS} neighbour craters (it has {len(p['members'])})."); return
        if len(my_pats()) >= MAX_PATTERNS_PER_IMAGE: messagebox.showinfo("Pattern", f"Maximum {MAX_PATTERNS_PER_IMAGE} patterns per image. Delete one first."); return
        p["name"] = f"{Path(S['path']).stem}-P{len(my_pats()) + 1}"; S["patterns"].append(p); S["props"] = [q for q in S["props"] if q is not p]
        S["store"].save_patterns(S["patterns"]); d = S["store"].out / Path(S["path"]).stem; d.mkdir(exist_ok=True); cv2.imwrite(str(d / "patterns.png"), draw_patterns_overlay(S["img"], my_pats()))
        say(f"Pattern '{p['name']}' saved ({len(p['members']) + 1} craters)  ->  {S['store'].out / 'patterns.json'}"); S["cur"] = S["props"][0] if S["props"] else p; pstatus(); redraw()

    def edit_pat():
        p = S["cur"]
        if p is None: return
        if is_saved(p): S["patterns"] = [q for q in S["patterns"] if q is not p]; p["name"] = "edited"; S["props"].insert(0, p); S["store"].save_patterns(S["patterns"]); say("Pattern opened for editing (not saved until you press OK).")
        tool.set("pat member"); say("EDIT: tool 'pat anchor' = click a crater to make it the anchor | tool 'pat member' = click a crater to add/remove it, or DRAG to draw a new crater and add it. Then OK."); pstatus(); redraw()

    def new_pat():
        if S["img"] is None: return
        if len(my_pats()) >= MAX_PATTERNS_PER_IMAGE: messagebox.showinfo("Pattern", f"Maximum {MAX_PATTERNS_PER_IMAGE} patterns per image."); return
        p = dict(id=uuid.uuid4().hex[:8], name="new", image=S["path"], scale=S["scale"], anchor=None, members=[]); S["props"].append(p); S["cur"] = p
        tool.set("pat anchor"); say("NEW pattern: 1) tool 'pat anchor': click the big central crater  2) tool 'pat member': click 3-7 neighbour craters (drag = draw a new crater)  3) OK."); pstatus(); redraw()

    def del_pat():
        p = S["cur"]
        if p is None: return
        if is_saved(p):
            if len(my_pats()) == 1 and not messagebox.askyesno("Delete", "This is the only pattern of this image (1 is required). Delete anyway?"): return
            S["patterns"] = [q for q in S["patterns"] if q is not p]; S["store"].save_patterns(S["patterns"])
        else: S["props"] = [q for q in S["props"] if q is not p]
        v = view_list(); S["cur"] = v[0] if v else None; pstatus(); redraw()

    def pat_click(x, y, kind):
        p = S["cur"]
        if p is not None and is_saved(p): messagebox.showinfo("Pattern", "This pattern is saved. Press 'Edit' first to change it."); return False
        if p is None: new_pat(); p = S["cur"]
        c = pick(x, y)
        if c is None: say("No crater there. Draw it first with the 'crater' tool, or (tool 'pat member') drag to draw it."); return False
        if kind == "anchor":
            p["anchor"] = _c3(c); p["members"] = [m for m in p["members"] if not same_crater(m, c)]; tool.set("pat member")
        else:
            if p.get("anchor") is None: say("Pick the anchor first (tool 'pat anchor')."); return False
            if same_crater(p["anchor"], c): return False
            hit = next((m for m in p["members"] if same_crater(m, c)), None)
            if hit is not None: p["members"].remove(hit)
            elif len(p["members"]) >= MAX_MEMBERS: say(f"Maximum {MAX_MEMBERS} neighbours."); return False
            else: p["members"].append(_c3(c))
        p["ambiguous"] = False; pstatus(); redraw(); return True

    def probe_for(img):
        if S["clf"] is None: return None
        F = FeatPyr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)); return lambda circ: S["clf"].predict_proba(F(circ.astype(np.float32)))[:, 1]

    def low_pool_for(img, cands):
        if S["clf"] is not None: d = detect_ml(img, S["clf"], 0.25, args.min_r, args.max_r, cands, log=lambda s: None)
        else: d = [dict(cx=c["cx"], cy=c["cy"], r=math.sqrt(c["a"] * c["b"])) for c in classic_candidates(img, args.min_r, args.max_r)[1]["final"]]
        return d

    def locate_here():
        if S["img"] is None or not S["patterns"]: messagebox.showinfo("Locate", "Save at least one pattern first."); return
        if S["cands"] is None: classic()
        low = pool(True)
        if S["clf"] is not None: low += [(d["cx"], d["cy"], d["r"]) for d in detect_ml(S["img"], S["clf"], 0.25, args.min_r, args.max_r, S["cands"], log=lambda s: None) if not any(same_crater((d["cx"], d["cy"], d["r"]), c) for c in low)]
        pr = probe_for(S["img"]); S["found"] = []
        for p in S["patterns"]:
            r = find_pattern(p, low, probe=pr, frame=S["img"].shape[:2])
            say(f"  {p['name']}: " + ("not found" if r is None else f"{'FOUND' if r['found'] else 'weak'}  craters {r['n_det']} detected + {r['n_probed']} pattern-guided of {r['n']}, zoom x{r['scale']:.3f}, rotation {r['rotation_deg']:.1f} deg"))
            if r and r["found"]: S["found"].append((p, r))
        redraw()

    def locate_other():
        if not S["patterns"]: messagebox.showinfo("Locate", "Save at least one pattern first."); return
        pth = filedialog.askopenfilename(title="Image to search the saved patterns in (other zoom / angle / sun)", filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
        if not pth: return
        imgB, scB = load_img(pth, args.max_side); say(f"Searching {len(S['patterns'])} saved pattern(s) in {Path(pth).name} ...")
        cB = classic_candidates(imgB, args.min_r, args.max_r)[0]; lowB = [(d["cx"], d["cy"], d["r"]) for d in low_pool_for(imgB, cB)]; pr = probe_for(imgB); res = []
        for p in S["patterns"]:
            r = find_pattern(p, lowB, probe=pr, frame=imgB.shape[:2]); res.append((p, r))
            say(f"  {p['name']}: " + ("not found" if r is None else f"{'FOUND' if r['found'] else 'weak'}  {r['n_det']} detected + {r['n_probed']} pattern-guided of {r['n']}, zoom x{r['scale']:.3f}, rotation {r['rotation_deg']:.1f} deg"))
        found = [(p, r) for p, r in res if r and r["found"]]; d = S["store"].out / f"pattern_search_{Path(pth).stem}"; d.mkdir(exist_ok=True)
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pattern", "source_image", "found", "n_pattern_craters", "n_detected", "n_pattern_guided", "score", "zoom", "rotation_deg", "anchor_x", "anchor_y"])
            for p, r in res: w.writerow([p["name"], p["image"], bool(r and r["found"]), p.get("members") and len(p["members"]) + 1, r and r["n_det"], r and r["n_probed"], r and round(r["score"], 3), r and round(r["scale"], 4), r and round(r["rotation_deg"], 2), r and round(r["members"][0]["cx"], 2), r and round(r["members"][0]["cy"], 2)])
        ov = draw_patterns_overlay(imgB, [], found); cv2.imwrite(str(d / "overlay.png"), ov); say(f"{len(found)} of {len(res)} patterns found -> {d}")
        if found and messagebox.askyesno("Review", f"{len(found)} pattern(s) recognised in {Path(pth).name}.\nOpen that image and review/edit/save them as ITS patterns?"):
            open_image(pth); props = []
            for p, r in found:
                ms = r["members"]; q = dict(id=uuid.uuid4().hex[:8], name=f"found {p['name']}", image=S["path"], scale=S["scale"], anchor=_c3((ms[0]["cx"], ms[0]["cy"], ms[0]["r"])),
                                            members=[_c3((m["cx"], m["cy"], m["r"])) for m in ms[1:]], ambiguous=False)
                props.append(q)
            S["props"] = props[:MAX_PATTERNS_PER_IMAGE]; S["cur"] = S["props"][0]; pstatus(); redraw()

    def final_list():
        man = [refine_ellipse(Feat(cv2.cvtColor(S["img"], cv2.COLOR_BGR2GRAY)), a["cx"], a["cy"], a["r"]) for a in mine() if a["label"] == 1]
        for m in man: m.update(conf=1.0, source="manual")
        neg = [(a["cx"], a["cy"], a["r"]) for a in mine() if a["label"] == 0]
        ml = [d for d in S["dets"] if d["source"] == "ml" and d["conf"] >= thr.get() and not any(same_crater((d["cx"], d["cy"], d["r"]), n) for n in neg)
              and not any(same_crater((d["cx"], d["cy"], d["r"]), (m["cx"], m["cy"], m["r"])) for m in man)]
        return sorted(man + ml, key=lambda d: -d["r"])

    def save_all():
        if S["img"] is None: return
        if not my_pats():
            r = messagebox.askyesnocancel("Pattern needed", f"This image has no pattern yet (1 to {MAX_PATTERNS_PER_IMAGE} patterns per image are required).\n\nYes = propose patterns now      No = save craters anyway      Cancel = go back")
            if r is None: return
            if r: propose(); return
        sync(); dets = final_list(); d = S["store"].save_craters(S["path"], dets, S["img"], S["scale"], S["img"].shape); say(f"Saved {len(dets)} craters ({sum(x['source'] == 'manual' for x in dets)} yours + {sum(x['source'] == 'ml' for x in dets)} ML) -> {d}")

    def match():
        if S["img"] is None or S["clf"] is None: messagebox.showwarning("Need", "Open image A and train/load a model first."); return
        p = filedialog.askopenfilename(title="Second image (different light / angle)", filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
        if not p: return
        imgB, scB = load_img(p, args.max_side); say("Detecting craters in image B with the same model ...")
        cB, _ = classic_candidates(imgB, args.min_r, args.max_r); dB = detect_ml(imgB, S["clf"], thr.get(), args.min_r, args.max_r, cB, log=say); dA = final_list()
        m = match_patterns(dA, dB)
        if m is None: say("No consistent crater pattern found (need >= 4 matching craters). Add labels / lower the confidence."); return
        d, canvas = save_match(S["store"], S["path"], p, S["img"], imgB, m)
        say(f"PATTERN MATCH: {m['n_inliers']} craters matched, scale {m['scale']:.3f}, rotation {m['rotation_deg']:.2f} deg, RMSE {m['rmse_px']:.2f} px -> {d}")
        top = tk.Toplevel(root); top.title("Pattern match (red lines = same crater)"); h, w = canvas.shape[:2]; k = min(1.0, 1300 / w, 800 / h)
        ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(canvas, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))); lab = tk.Label(top, image=ph); lab.image = ph; lab.pack()

    def pick_out():
        d = filedialog.askdirectory()
        if d: S["store"] = Store(d); S["ann"] = S["store"].load_ann(); S["clf"] = S["store"].load_model(); say(f"Output folder: {d}  ({len(S['ann'])} labels loaded, model {'found' if S['clf'] else 'none'})"); redraw()

    for t, c in (("Open image", open_image), ("Classic detect", classic), ("Auto-label", autolabel), ("Train ML", train), ("Self-train", selftrain), ("Detect (ML)", detect), ("Save craters", save_all), ("Match pattern...", match), ("Output folder...", pick_out)):
        ttk.Button(bar, text=t, command=c).pack(side="left", padx=2)
    ttk.Label(bar2, text="Tool:").pack(side="left")
    for t in ("crater", "NOT crater", "accept", "reject", "erase"): ttk.Radiobutton(bar2, text=t, variable=tool, value=t).pack(side="left", padx=4)
    ttk.Label(bar2, text="   ML confidence >=").pack(side="left"); tk.Scale(bar2, from_=0.05, to=0.95, resolution=0.05, orient="horizontal", length=140, variable=thr, command=lambda v: redraw()).pack(side="left")
    ttk.Checkbutton(bar2, text="this image is FULLY labelled (un-drawn = not crater)", variable=complete, command=sync).pack(side="left", padx=10)
    ttk.Checkbutton(bar2, text="show detections", variable=show_dets, command=redraw).pack(side="left")
    ttk.Label(bar2, text="   wheel = zoom,  Shift+drag = pan").pack(side="left")
    ttk.Label(bar3, text="PATTERNS:").pack(side="left")
    for t, c in (("Propose patterns", propose), ("<", lambda: step(-1)), (">", lambda: step(1)), ("OK - save", ok_pat), ("Edit", edit_pat), ("New (draw)", new_pat), ("Delete", del_pat), ("Locate here", locate_here), ("Locate in other image...", locate_other)):
        ttk.Button(bar3, text=t, command=c, width=max(3, len(t))).pack(side="left", padx=2)
    for t in ("pat anchor", "pat member"): ttk.Radiobutton(bar3, text=t, variable=tool, value=t).pack(side="left", padx=4)
    ttk.Label(bar4, textvariable=pstat, foreground="#06c", font=("Arial", 11, "bold")).pack(side="left")
    S["ann"] = S["store"].load_ann(); say(f"Output folder: {S['store'].out.resolve()}  | {len(S['ann'])} saved labels | model: {'loaded' if S['clf'] else 'none yet'}")
    S["_api"] = dict(S=S, locate_other=locate_other, propose=propose, ok=ok_pat, edit=edit_pat, newpat=new_pat, delpat=del_pat, click=pat_click, locate=locate_here, step=step, autolabel=autolabel, selftrain=selftrain, open=open_image, classic=classic, train=train, detect=detect, save=save_all, add=add_ann, root=root)
    if args.image: root.after(300, lambda: open_image(args.image))
    root.mainloop()


# =============================================================== self-test on synthetic terrain
def synth_terrain(seed=0, n=45, size=900, hard=True):
    rng = np.random.RandomState(seed); yy, xx = np.mgrid[0:size, 0:size].astype(np.float32); h = np.zeros((size, size), np.float32); GT = []
    for _ in range(n * 3):
        if len(GT) >= n: break
        r = float(np.exp(rng.uniform(math.log(12), math.log(70)))); cx, cy = rng.uniform(r, size - r, 2)
        if any(math.hypot(cx - g[0], cy - g[1]) < 0.9 * (r + g[2]) for g in GT): continue
        GT.append((float(cx), float(cy), r)); rho = np.hypot(xx - cx, yy - cy) / r; d = 0.22 * r * (rng.uniform(0.10, 1.0) if hard else 1.0)   # old, eroded craters are shallow
        h += np.where(rho < 1, -d * (1 - rho ** 2), 0.18 * d * np.exp(-((rho - 1) / 0.18) ** 2) + 0.05 * d * np.exp(-(rho - 1) / 0.7)) * (rho < 2.5)
    for s, amp in ((2, 0.9), (6, 2.0), (20, 3.5)): h += cv2.GaussianBlur(rng.randn(size, size).astype(np.float32), (0, 0), s) * amp * s ** 0.5
    return h, GT


def shade(h, az, el, k=2.0):
    hx, hy = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8 * k, cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8 * k
    n = np.stack([-hx, -hy, np.ones_like(hx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)), math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
    return np.clip(0.15 + 0.85 * np.clip(n @ L, 0, 1), 0, 1)


def score_dets(dets, GT, tol=0.25, ALL=None):
    hit, errs, used = 0, [], set()
    for g in GT:
        c = [(math.hypot(d["cx"] - g[0], d["cy"] - g[1]), i) for i, d in enumerate(dets) if i not in used and abs(d["r"] / g[2] - 1) < 0.3]
        if c and min(c)[0] < tol * g[2]: hit += 1; used.add(min(c)[1]); errs.append(min(c)[0])
    if ALL is not None:                              # extra = detections that match NO true crater at all
        extra = sum(1 for d in dets if not any(math.hypot(d["cx"] - g[0], d["cy"] - g[1]) < tol * g[2] and abs(d["r"] / g[2] - 1) < 0.3 for g in ALL))
        return hit, extra, errs
    return hit, len(dets) - len(used), errs


def selftest():
    rng = np.random.RandomState(1); size = 900; h, GT = synth_terrain(0, 45, size)
    A = (shade(h, 40, 25) * 255 + np.random.RandomState(2).randn(size, size) * 7).clip(0, 255).astype(np.uint8)
    th, s, t = math.radians(20), 0.8, np.array([60.0, 90.0]); M = np.array([[s * math.cos(th), -s * math.sin(th), t[0]], [s * math.sin(th), s * math.cos(th), t[1]]], np.float32)
    hB = cv2.warpAffine(h, M, (size, size), flags=cv2.INTER_CUBIC) * s                   # same terrain, rotated + scaled ...
    B = (shade(hB, 200, 38) * 255 + np.random.RandomState(3).randn(size, size) * 7).clip(0, 255).astype(np.uint8)   # ... and lit from the OTHER side, higher sun
    GTB = [(M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2], s * r) for x, y, r in GT]
    imgA, imgB = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR), cv2.cvtColor(B, cv2.COLOR_GRAY2BGR); print(f"synthetic terrain: {len(GT)} craters, image B = rotated {math.degrees(th):.0f} deg, scale {s}, sun moved ~160 deg")
    cands, res = classic_candidates(imgA, 10, 100); hit, fp, _ = score_dets(res["final"], GT, ALL=GT)
    classic_all = res["final"]; print(f"1. classic crater_lens pipeline (defaults):  found {hit}/{len(GT)}   false {fp}")
    order = rng.permutation(len(GT)); lab = [GT[i] for i in order[:10]]; testset = [GT[i] for i in order[10:]]
    pos = [(x + rng.normal(0, .02 * r), y + rng.normal(0, .02 * r), r * (1 + rng.normal(0, .03))) for x, y, r in lab]
    neg = []
    while len(neg) < 6:
        x, y, r = rng.uniform(80, size - 80), rng.uniform(80, size - 80), rng.uniform(15, 50)
        if not any(math.hypot(x - g[0], y - g[1]) < 0.8 * (r + g[2]) for g in GT): neg.append((x, y, r))
    t0 = time.time(); clf, rep = train_model([dict(gray=A, pos=pos, neg=neg, complete=False, cands=cands)]); print(f"2. trained on {len(pos)} drawn craters + {len(neg)} NOT-craters in {time.time() - t0:.1f}s")
    dets = detect_ml(imgA, clf, 0.5, 10, 100, cands, log=lambda s: None); hit, fp, errs = score_dets(dets, testset, ALL=GT)
    hc, _, _ = score_dets(classic_all, testset)
    print(f"3. the {len(testset)} craters you did NOT draw:  classic found {hc}  ->  ML found {hit}   (extra detections: {fp})")
    raw = [dict(cx=g[0], cy=g[1], r=g[2], a=g[2], b=g[2], angle=0) for g in testset]
    print(f"   sub-pixel centre error (median) of found craters: {np.median(errs):.2f} px  (crater radius 12-70 px)")
    detsB = detect_ml(imgB, clf, 0.5, 8, 90, classic_candidates(imgB, 8, 90)[0], log=lambda s: None); hitB, fpB, errB = score_dets(detsB, GTB, ALL=GTB)
    print(f"4. SAME model on image B (other light + rotated + scaled), all {len(GTB)} craters:  found {hitB}/{len(GTB)}   extra {fpB}   centre err median {np.median(errB):.2f} px")
    m = match_patterns(detsB if False else dets, detsB)
    if m is None: print("5. pattern match FAILED"); return
    print(f"5. crater-PATTERN match A->B: {m['n_inliers']} craters matched, scale {m['scale']:.4f} (true {s}), rotation {m['rotation_deg']:.2f} deg (true {math.degrees(th):.0f}),"
          f" translation ({m['tx']:.1f},{m['ty']:.1f}) (true {t[0]:.0f},{t[1]:.0f}),  RMSE {m['rmse_px']:.2f} px")


def selftest_patterns():
    """patterns proposed on image A are searched in image B (rotated, scaled, lit from the other side) at several sun elevations."""
    rng = np.random.RandomState(1); size = 900; h, GT = synth_terrain(0, 45, size)
    A = (shade(h, 40, 25) * 255 + np.random.RandomState(2).randn(size, size) * 7).clip(0, 255).astype(np.uint8); imgA = cv2.cvtColor(A, cv2.COLOR_GRAY2BGR)
    th, s, t = math.radians(20), 0.8, np.array([60.0, 90.0]); M = np.array([[s * math.cos(th), -s * math.sin(th), t[0]], [s * math.sin(th), s * math.cos(th), t[1]]], np.float32)
    hB = cv2.warpAffine(h, M, (size, size), flags=cv2.INTER_CUBIC) * s
    cands = classic_candidates(imgA, 10, 100)[0]; order = rng.permutation(len(GT))
    pos = [(x + rng.normal(0, .02 * r), y + rng.normal(0, .02 * r), r) for x, y, r in [GT[i] for i in order[:10]]]
    t0 = time.time(); clf, _ = train_model([dict(gray=A, pos=pos, neg=[], complete=False, cands=cands)]); print(f"trained in {time.time() - t0:.1f}s")
    dA = detect_ml(imgA, clf, 0.5, 10, 100, cands, log=lambda s: None); pats = propose_patterns([(d["cx"], d["cy"], d["r"]) for d in dA], 4, 5)
    print(f"A: {len(dA)} craters -> {len(pats)} proposed patterns:", [(len(p['members']) + 1, p['quality'], 'AMBIGUOUS' if p['ambiguous'] else 'unique') for p in pats])
    sA = synth_terrain  # noqa
    for el in (10, 25, 45, 70):
        B = (shade(hB, 200, el) * 255 + np.random.RandomState(3).randn(size, size) * 7).clip(0, 255).astype(np.uint8); imgB = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
        cB = classic_candidates(imgB, 8, 90)[0]; dB = detect_ml(imgB, clf, 0.3, 8, 90, cB, log=lambda s: None); FB = FeatPyr(B)
        GTB = [(M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2], s * r) for x, y, r in GT]; hit, fp, _ = score_dets([d for d in dB if d["conf"] >= 0.5], GTB, ALL=GTB)
        probe = lambda circ: clf.predict_proba(FB(circ.astype(np.float32)))[:, 1]; ok = 0; errs = []
        for p in pats:
            r = find_pattern(p, [(d["cx"], d["cy"], d["r"]) for d in dB], probe=probe, frame=B.shape)
            if r and r["found"]:
                a = p["anchor"]; tp = apply_T(M.tolist() + [[0, 0, 1]] if False else np.vstack([M, [0, 0, 1]]), [a[:2]])[0]; got = apply_T(r["T"], [a[:2]])[0]
                errs.append((abs(r["scale"] / s - 1) * 100, abs(((r["rotation_deg"] - 20 + 180) % 360) - 180), float(np.hypot(*(tp - got))))); ok += 1
        e = np.array(errs) if errs else np.zeros((1, 3)) * np.nan
        print(f"B sun elevation {el:2d} deg: craters found {hit}/45 (extra {fp}) | patterns found {ok}/{len(pats)} | scale err {np.nanmean(e[:, 0]):.2f}%  rot err {np.nanmean(e[:, 1]):.2f} deg  anchor err {np.nanmean(e[:, 2]):.2f} px")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image"); ap.add_argument("--match", nargs=2, metavar=("A", "B")); ap.add_argument("--model")
    ap.add_argument("--out", default="output"); ap.add_argument("--max-side", type=int, default=1400); ap.add_argument("--min-r", type=float, default=7); ap.add_argument("--max-r", type=float, default=200)
    ap.add_argument("--conf", type=float, default=0.5); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--selftest-patterns", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    if a.selftest_patterns: return selftest_patterns()
    if a.match or (a.image and a.model):
        st = Store(a.out); clf = None
        if a.model: import joblib; clf = joblib.load(a.model)
        else: clf = st.load_model()
        if clf is None: raise SystemExit("need --model")
        if a.match:
            (iA, sA), (iB, sB) = load_img(a.match[0], a.max_side), load_img(a.match[1], a.max_side)
            dA = detect_ml(iA, clf, a.conf, a.min_r, a.max_r, classic_candidates(iA, a.min_r, a.max_r)[0]); dB = detect_ml(iB, clf, a.conf, a.min_r, a.max_r, classic_candidates(iB, a.min_r, a.max_r)[0])
            st.save_craters(a.match[0], dA, iA, sA, iA.shape); st.save_craters(a.match[1], dB, iB, sB, iB.shape); m = match_patterns(dA, dB)
            print("no consistent pattern" if m is None else f"matched {m['n_inliers']} craters, scale {m['scale']:.3f}, rot {m['rotation_deg']:.2f}, RMSE {m['rmse_px']:.2f}px -> {save_match(st, a.match[0], a.match[1], iA, iB, m)[0]}")
        else:
            img, sc = load_img(a.image, a.max_side); d = detect_ml(img, clf, a.conf, a.min_r, a.max_r, classic_candidates(img, a.min_r, a.max_r)[0]); print("saved ->", st.save_craters(a.image, d, img, sc, img.shape))
        return
    run_gui(a)


if __name__ == "__main__":
    main()
