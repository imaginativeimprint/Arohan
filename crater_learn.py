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
import crater_lens as CL

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


def build_training(items, rng):
    """items: list of dict(gray, pos(N,3), neg(M,3), complete(bool), cands(K,3)).  Returns X, y, w."""
    X, y, w = [], [], []
    for it in items:
        F = FeatPyr(it["gray"]); H, W = it["gray"].shape; pos, neg = np.asarray(it["pos"], np.float32).reshape(-1, 3), np.asarray(it["neg"], np.float32).reshape(-1, 3)
        if len(pos): P = jitter(pos, rng, 5); X.append(F(P)); y += [1] * len(P); w += [1.0] * len(P)
        if len(neg): N_ = jitter(neg, rng, 3); X.append(F(N_)); y += [0] * len(N_); w += [1.0] * len(N_)
        cands = np.asarray(it.get("cands", np.zeros((0, 3))), np.float32).reshape(-1, 3)
        free = np.array([not any(same_crater(c, p, 0.7) for p in pos) and not any(same_crater(c, n, 0.7) for n in neg) for c in cands], bool) if len(cands) else np.zeros(0, bool)
        if free.any():
            cf = cands[free]
            if it["complete"]: sel, wt = cf[rng.permutation(len(cf))[:max(200, 4 * len(pos) * 6)]], 1.0     # fully labelled image: all unlabeled = NOT crater
            else: sel, wt = cf[rng.permutation(len(cf))[:max(30, 2 * len(pos) * 6)]], 0.3                    # partly labelled: weak negatives
            X.append(F(sel)); y += [0] * len(sel); w += [wt] * len(sel)
    if not X: return None, None, None
    return np.vstack(X), np.array(y), np.array(w)


def train_model(items, seed=0):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    rng = np.random.RandomState(seed); X, y, w = build_training(items, rng)
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
    S = dict(img=None, path="", scale=1.0, ann=[], dets=[], store=Store(args.out), clf=None, zoom=1.0, ox=0.0, oy=0.0, photo=None, cands=None, drag=None, pan=None)
    S["clf"] = S["store"].load_model(); proj = S["store"].load_proj()
    tool = tk.StringVar(value="crater"); thr = tk.DoubleVar(value=0.5); complete = tk.BooleanVar(value=False); show_dets = tk.BooleanVar(value=True)
    bar = ttk.Frame(root, padding=3); bar.pack(fill="x"); bar2 = ttk.Frame(root, padding=3); bar2.pack(fill="x")
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

    def open_image(p=None):
        p = p or filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
        if not p: return
        S["img"], S["scale"] = load_img(p, args.max_side); S["path"] = os.path.abspath(p); S["dets"] = []; S["cands"] = None
        H, W = S["img"].shape[:2]; cw, ch = max(cv.winfo_width(), 50), max(cv.winfo_height(), 50); S["zoom"] = min(cw / W, ch / H); S["ox"] = S["oy"] = 0
        complete.set(bool(proj["complete"].get(S["path"], False))); say(f"Opened {p}  ({W}x{H}, scale to original = {S['scale']:.3f})  | your labels here: {len(mine())}"); redraw()

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
        if t in ("crater", "NOT crater"): S["drag"] = ((e.x, e.y), (e.x, e.y))
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
            if r >= 3: cx, cy = c2i(xa, ya); add_ann(cx, cy, r, 1 if tool.get() == "crater" else 0)
            redraw()

    def wheel(e):
        f = 1.25 if (getattr(e, "delta", 0) > 0 or e.num == 4) else 0.8; x, y = c2i(e.x, e.y); S["zoom"] *= f; S["ox"] = e.x - x * S["zoom"]; S["oy"] = e.y - y * S["zoom"]; redraw()
    cv.bind("<ButtonPress-1>", press); cv.bind("<B1-Motion>", move); cv.bind("<ButtonRelease-1>", release); cv.bind("<MouseWheel>", wheel); cv.bind("<Button-4>", wheel); cv.bind("<Button-5>", wheel); cv.bind("<Configure>", redraw)

    def final_list():
        man = [refine_ellipse(Feat(cv2.cvtColor(S["img"], cv2.COLOR_BGR2GRAY)), a["cx"], a["cy"], a["r"]) for a in mine() if a["label"] == 1]
        for m in man: m.update(conf=1.0, source="manual")
        neg = [(a["cx"], a["cy"], a["r"]) for a in mine() if a["label"] == 0]
        ml = [d for d in S["dets"] if d["source"] == "ml" and d["conf"] >= thr.get() and not any(same_crater((d["cx"], d["cy"], d["r"]), n) for n in neg)
              and not any(same_crater((d["cx"], d["cy"], d["r"]), (m["cx"], m["cy"], m["r"])) for m in man)]
        return sorted(man + ml, key=lambda d: -d["r"])

    def save_all():
        if S["img"] is None: return
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
    S["ann"] = S["store"].load_ann(); say(f"Output folder: {S['store'].out.resolve()}  | {len(S['ann'])} saved labels | model: {'loaded' if S['clf'] else 'none yet'}")
    S["_api"] = dict(autolabel=autolabel, selftrain=selftrain, open=open_image, classic=classic, train=train, detect=detect, save=save_all, add=add_ann, root=root)
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image"); ap.add_argument("--match", nargs=2, metavar=("A", "B")); ap.add_argument("--model")
    ap.add_argument("--out", default="output"); ap.add_argument("--max-side", type=int, default=1400); ap.add_argument("--min-r", type=float, default=7); ap.add_argument("--max-r", type=float, default=200)
    ap.add_argument("--conf", type=float, default=0.5); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
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
