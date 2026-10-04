#!/usr/bin/env python3
"""
Crater Register v2  -  register two Chandrayaan-2 / LRO images by their CRATER PATTERN (zoom, rotation, sun angle independent)
===========================================================================================================================
Built on crater_learn.py (same folder; it also needs your crater_lens.py):
    pip install numpy opencv-python pillow scipy scikit-learn joblib      (matplotlib optional: PDF report)

    python crater_register.py                                   # GUI
    python crater_register.py --ref A.png --tgt B.png --out output [--ref-xml A.xml --tgt-xml B.xml --ref-csv A_grid.csv --tgt-csv B_grid.csv]

What it does
  1 craters: from your trained model (output/model.joblib), or crater CSVs, or auto-trained on the pair
  2 pattern: local crater "lock patterns" (+ the patterns you saved in crater_learn) vote for zoom / rotation / shift
  3 verify : the transform that lines up the most whole craters wins (craters outside the other frame are not counted against it)
  4 fit    : similarity / affine / homography (auto by leave-one-out error), optional dense sub-pixel refinement (ECC)
  5 outputs: registered image, overlay, checkerboard, match points CSV, metrics (RMSE, held-out RMSE, inliers, inlier ratio,
             coverage / uniformity), lon/lat grid CSV for the target (if the reference has a grid CSV), report (HTML + PDF)
  6 viewer : both images side by side + a seek bar: slide / play and the target glides onto the reference by the pattern
"""
import argparse, base64, csv, html, io, itertools, json, math, os, re, sys, time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crater_learn as CLN
CL = CLN.CL

R_MOON = 1737400.0
OUT = Path("output")
DEFAULTS = dict(max_side=1400, min_r=8, max_frac=0.35, max_pattern=80, k=8, tol=0.05, max_craters=160, ml_p=0.5, model="auto",
                use_ecc=True, use_library=True, use_model=True, baseline=True, stress=False, make_report=True, use_csv_craters=True)


# =============================================================== XML label (PDS4) -> facts
def _num(v):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v)); return float(m.group()) if m else None


def parse_xml(path):
    """Reads any PDS4-style label.  facts = the useful items (times, instrument, size, sun / viewing angles when present);
    table = every leaf (key, value) for the report appendix."""
    info = dict(file=str(path or ""), facts={}, table=[])
    if not path: return info
    try: root = ET.parse(path).getroot()
    except Exception as ex: info["error"] = f"{type(ex).__name__}: {ex}"; return info
    rows = []
    def walk(el, pth):
        tag = el.tag.split("}")[-1]; p = pth + [tag]; txt = (el.text or "").strip()
        if len(el) == 0 and txt: rows.append(("/".join(p[-3:]), txt))
        for c in el: walk(c, p)
    walk(root, []); info["table"] = rows
    def find(*pats):
        for k, v in rows:
            if any(re.search(p, k, re.I) for p in pats): return v
    f = info["facts"]
    f["product_id"] = find(r"logical_identifier")
    f["title"] = find(r"(^|/)title$")
    f["start_time"] = find(r"start_date_time", r"start_time", r"image_start")
    f["stop_time"] = find(r"stop_date_time", r"stop_time", r"image_stop")
    f["instrument"] = next((v for k, v in rows if re.search(r"Observing_System_Component/name$", k) and "chandrayaan" not in v.lower() and "orbiter" in v.lower() or re.search(r"instrument.*name", k, re.I)), None)
    f["mission"] = find(r"Investigation_Area/name", r"mission")
    f["target"] = find(r"Target_Identification/name")
    lines = samples = None
    for i, (k, v) in enumerate(rows):                                   # Axis_Array: axis_name Line/Sample then elements
        if k.endswith("axis_name") and i + 1 < len(rows) and rows[i + 1][0].endswith("elements"):
            n = _num(rows[i + 1][1]); lines = n if v.lower().startswith("line") else lines; samples = n if v.lower().startswith("sample") else samples
    if lines: f["lines"] = int(lines)
    if samples: f["samples"] = int(samples)
    for key, pats in (("sun_elevation_deg", (r"sun.*elev", r"solar.*elev")), ("sun_azimuth_deg", (r"sun.*azim", r"solar.*azim")), ("incidence_angle_deg", (r"incidence",)),
                      ("emission_angle_deg", (r"emission",)), ("phase_angle_deg", (r"phase_angle", r"phase")), ("altitude_km", (r"altitude", r"spacecraft.*dist")),
                      ("resolution_m", (r"resolution", r"gsd", r"pixel_size", r"pixel_scale")), ("orbit", (r"orbit_number", r"orbit_no", r"(^|/)orbit$"))):
        v = find(*pats); n = _num(v) if v is not None else None
        if n is not None: f[key] = n
    return {**info, "facts": {k: v for k, v in f.items() if v is not None}}


def _t(s):
    try: return datetime.fromisoformat(s.replace("Z", "").replace("z", ""))
    except Exception: return None


# =============================================================== geo grid CSV (ISRO _g_grd_ format)
class Stereo:
    def __init__(self, lon0, lat0): self.l0, self.p0 = math.radians(lon0), math.radians(lat0)
    def fwd(self, lon, lat):
        l, p = np.radians(lon) - self.l0, np.radians(lat); k = 2 / (1 + math.sin(self.p0) * np.sin(p) + math.cos(self.p0) * np.cos(p) * np.cos(l))
        return R_MOON * k * np.cos(p) * np.sin(l), R_MOON * k * (math.cos(self.p0) * np.sin(p) - math.sin(self.p0) * np.cos(p) * np.cos(l))
    def inv(self, x, y):
        rho = np.hypot(x, y); c = 2 * np.arctan2(rho, 2 * R_MOON); rs = np.where(rho == 0, 1, rho)
        p = np.arcsin(np.clip(np.cos(c) * math.sin(self.p0) + y * np.sin(c) * math.cos(self.p0) / rs, -1, 1))
        l = self.l0 + np.arctan2(x * np.sin(c), rho * math.cos(self.p0) * np.cos(c) - y * math.sin(self.p0) * np.sin(c)); return np.degrees(l) % 360, np.degrees(p)


class CsvGeo:
    """grid CSV (Longitude,Latitude,Pixel,Scan) of an image whose ORIGINAL size is (W,H).  Work pixels (the loaded, possibly shrunk
    copy; work = original * scale) are converted to original pixels with the pixel-centre convention."""
    def __init__(self, path, orig_size, scale):
        d = pd_read(path); pix, scan = np.sort(d["Pixel"].unique()), np.sort(d["Scan"].unique()); d = d.sort_values(["Scan", "Pixel"])
        lon = d["Longitude"].values.reshape(len(scan), len(pix)); lat = d["Latitude"].values.reshape(len(scan), len(pix))
        self.st = Stereo(float(np.degrees(np.angle(np.exp(1j * np.radians(lon)).mean())) % 360), float(lat.mean())); X, Y = self.st.fwd(lon, lat)
        self.sx, self.sy = (pix.max() + 1) / orig_size[0], (scan.max() + 1) / orig_size[1]; self.scale = scale
        self.full_size = (int(pix.max()) + 1, int(scan.max()) + 1)
        self._f = RegularGridInterpolator((scan, pix), np.stack([X, Y], -1), bounds_error=False)
    def work2ll(self, x, y):
        xo, yo = (np.asarray(x, float) + .5) / self.scale - .5, (np.asarray(y, float) + .5) / self.scale - .5
        v = self._f(np.column_stack([(yo + .5) * self.sy - .5, (xo + .5) * self.sx - .5])); return self.st.inv(v[:, 0], v[:, 1])
    def gsd_work(self, shape):
        h, w = shape[:2]; lo, la = self.work2ll(np.array([w / 2, w / 2 + 10.0]), np.array([h / 2, h / 2])); x, y = self.st.fwd(lo, la); return float(np.hypot(x[1] - x[0], y[1] - y[0]) / 10)


def pd_read(path):
    import pandas as pd
    d = pd.read_csv(path); d.columns = [c.strip().capitalize() if c.strip().lower() in ("longitude", "latitude", "pixel", "scan") else c.strip() for c in d.columns]
    miss = [c for c in ("Longitude", "Latitude", "Pixel", "Scan") if c not in d.columns]
    if miss: raise ValueError(f"{path}: grid CSV needs columns Longitude, Latitude, Pixel, Scan (missing {miss})")
    return d


# =============================================================== crater CSV (optional input)  ->  array [cx,cy,a,b,angle,conf]
def load_crater_csv(path, scale):
    import pandas as pd
    d = pd.read_csv(path); d.columns = [c.strip().lower() for c in d.columns]; out = []
    for _, r in d.iterrows():
        if "cx_orig" in d.columns and "r_orig" in d.columns and r["r"] > 0:                  # produced by crater_learn: re-scale to our work size
            f = r["r_orig"] * scale / r["r"]; cx, cy = (r["cx_orig"] + .5) * scale - .5, (r["cy_orig"] + .5) * scale - .5
            a, b = (r["a"] * f, r["b"] * f) if "a" in d.columns else (r["r"] * f, r["r"] * f); ang = r.get("angle_deg", 0.0)
        else:
            cx, cy = r["cx"], r["cy"]; rr = r["r"] if "r" in d.columns else r.get("radius", 10); a, b, ang = r.get("a", rr), r.get("b", rr), r.get("angle_deg", 0.0)
        out.append([cx, cy, a, b, ang, r.get("confidence", 1.0) if "confidence" in d.columns else 1.0])
    return np.array(out, np.float64).reshape(-1, 6)


# =============================================================== small geometry helpers
def radius(P): return np.sqrt(P[:, 2] * P[:, 3])
def apply_T(T, P):
    P = np.asarray(P, float).reshape(-1, 2); q = np.c_[P, np.ones(len(P))] @ np.asarray(T).T; return q[:, :2] / q[:, 2:3]
def T_from_sim(s, th, t): c, si = math.cos(th), math.sin(th); return np.array([[s * c, -s * si, t[0]], [s * si, s * c, t[1]], [0, 0, 1.0]])
def sim_params(T): return math.hypot(T[0, 0], T[1, 0]), math.atan2(T[1, 0], T[0, 0])
def hstack_pad(a, b, gap=8):
    h = max(a.shape[0], b.shape[0]); o = np.full((h, a.shape[1] + b.shape[1] + gap, 3), 28, np.uint8); o[:a.shape[0], :a.shape[1]] = a; o[:b.shape[0], a.shape[1] + gap:] = b; return o
def tag(img, text, y=22):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA); cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); return img
def draw_ell(img, c, col, w=2): CL.draw_ellipse(img, float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), col, w)
def draw_set(img, C, col, w=2, ids=False):
    v = img.copy()
    for i, c in enumerate(C):
        draw_ell(v, c, col, w)
        if ids: cv2.putText(v, str(i), (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return v


# =============================================================== 1. crater detection (learn's ML), with optional CSV craters
def _compat_detect(img, clf, O, cand, log):
    try: return CLN.detect_ml(img, clf, float(O.get("ml_min", 0.25)), float(O["min_r"]) * 0.75, 0.4 * min(img.shape[:2]), cand, log=log)
    except Exception as ex: log(f"  (model not usable here: {type(ex).__name__}) "); return None


def detect_pair(ref_path, tgt_path, O, log=print):
    """returns dict with images, scales, original sizes, craters (n,6) and notes.  Crater CSVs skip detection for that image."""
    imgA, scA = CLN.load_img(ref_path, O["max_side"]); imgB, scB = CLN.load_img(tgt_path, O["max_side"])
    oA, oB = cv2.imread(ref_path).shape[:2][::-1], cv2.imread(tgt_path).shape[:2][::-1]; notes = []; res = dict(imgA=imgA, imgB=imgB, scA=scA, scB=scB, oA=oA, oB=oB, notes=notes)
    CA = load_crater_csv(O["ref_craters"], scA) if O.get("ref_craters") and O.get("use_csv_craters", True) else None
    CB = load_crater_csv(O["tgt_craters"], scB) if O.get("tgt_craters") and O.get("use_csv_craters", True) else None
    if CA is not None: notes.append(f"reference craters taken from CSV ({len(CA)})")
    if CB is not None: notes.append(f"target craters taken from CSV ({len(CB)})")
    rmin = float(O["min_r"]) * 0.75; rmaxf = lambda im: 0.4 * min(im.shape[:2])
    cand = {}
    if CA is None or CB is None:
        cand["A"] = CLN.classic_candidates(imgA, rmin, rmaxf(imgA))[0]; cand["B"] = CLN.classic_candidates(imgB, rmin, rmaxf(imgB))[0]
        clf = CLN.Store(O.get("out", OUT)).load_model() if O.get("use_model", True) else None
        if clf is not None:
            outs = [(_compat_detect(im, clf, O, cand[k], log) if C is None else None) for im, k, C in ((imgA, "A", CA), (imgB, "B", CB))]
            if any(o is None and C is None for o, C in zip(outs, (CA, CB))): clf = None; notes.append("saved model could not be used (old version?) -> auto-training")
            else: notes.append("detector: your trained model (output/model.joblib)")
        if clf is None:
            log("auto-training on the strong craters of the pair (+ your crater CSVs if given) ...")
            pos = lambda img, C, k: [(c[0], c[1], math.sqrt(c[2] * c[3])) for c in C] if C is not None else CLN.auto_label(img, rmin, rmaxf(img))[0]
            clf, rep = CLN.train_model([dict(gray=cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY), pos=pos(imgA, CA, "A"), neg=[], complete=False, cands=cand["A"]),
                                        dict(gray=cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY), pos=pos(imgB, CB, "B"), neg=[], complete=False, cands=cand["B"])])
            notes.append(f"detector: auto-trained on this pair ({rep['n_positive']} crater samples)"); outs = [_compat_detect(im, clf, O, cand[k], log) if C is None else None for im, k, C in ((imgA, "A", CA), (imgB, "B", CB))]
        res["clf"] = clf
        if CA is None: CA = np.array([[c["cx"], c["cy"], c["a"], c["b"], c["angle"], c["conf"]] for c in outs[0]], np.float64).reshape(-1, 6)
        if CB is None: CB = np.array([[c["cx"], c["cy"], c["a"], c["b"], c["angle"], c["conf"]] for c in outs[1]], np.float64).reshape(-1, 6)
    else: res["clf"] = None
    res.update(CA=CA, CB=CB, cand=cand); return res


def select_pattern(C, shape, min_r, max_frac, max_n):
    """medium craters that are not cut by the image border (a cut crater has a biased centre and size)"""
    h, w = shape[:2]; r = radius(C); ok = (r >= min_r) & (r <= max_frac * min(h, w)); m = 0.7 * r
    ok &= (C[:, 0] - m >= 0) & (C[:, 0] + m <= w) & (C[:, 1] - m >= 0) & (C[:, 1] + m <= h)
    idx = np.nonzero(ok)[0]; return idx[np.argsort(-r[idx])][:int(max_n)]


# =============================================================== 2. pattern = local lock-pattern triangles (anchor + 2 of its k nearest neighbours)
def knn_triangles(P, R, k):
    n = len(P)
    if n < 3: return None
    k = min(int(k), n - 1); _, nn = cKDTree(P).query(P, k + 1); tri = []
    for i in range(n):
        for a, b in itertools.combinations(nn[i, 1:], 2): tri.append((i, a, b))
    tri = np.unique(np.sort(np.array(tri), axis=1), axis=0); p = P[tri]
    d = lambda a, b: np.linalg.norm(p[:, a] - p[:, b], axis=1); sides = np.stack([d(1, 2), d(0, 2), d(0, 1)], 1)       # side opposite vertex 0, 1, 2
    order = np.argsort(sides, axis=1); verts = np.take_along_axis(tri, order, 1); ss = np.sort(sides, axis=1); L = ss[:, 2]
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]; area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    good = (ss[:, 0] / L > 0.22) & (area / L ** 2 > 0.05)
    return np.c_[ss[:, 0] / L, ss[:, 1] / L, R[verts] / L[:, None]][good], verts[good]


def fit_T(model, A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    if model == "similarity" or len(A) < 3:
        r = CLN.similarity_from(A, B); return None if r is None else T_from_sim(r[0], r[1], r[2])
    if model == "affine" or len(A) < 4:
        X = np.c_[A, np.ones(len(A))]; sol, *_ = np.linalg.lstsq(X, B, rcond=None); return np.vstack([sol.T, [0, 0, 1]])
    H, _ = cv2.findHomography(A.astype(np.float32), B.astype(np.float32), 0); return H


MIN_PTS = {"similarity": 2, "affine": 3, "homography": 4}


def consensus(T, PA, RA, PB, RB, treeB, thr, rtol=(0.72, 1.39)):
    """pairs (i,j): reference crater i lands on target crater j (position AND size agree); each target crater used once"""
    q = apply_T(T, PA[:, :2]); s = math.sqrt(abs(np.linalg.det(T[:2, :2]))) if abs(T[2, 0]) + abs(T[2, 1]) < 1e-9 else float(np.sqrt(abs(np.linalg.det(T[:2, :2])) / max(1e-9, (T[2, 0] * PA[:, 0] + T[2, 1] * PA[:, 1] + T[2, 2]).mean() ** 2)))
    d, nn = treeB.query(q); rr = RB[nn] / (RA * s); ok = (d < np.maximum(thr, 0.3 * RB[nn])) & (rr > rtol[0]) & (rr < rtol[1])
    best = {}
    for i in np.nonzero(ok)[0]:
        j = nn[i]
        if j not in best or d[i] < d[best[j]]: best[j] = i
    return sorted((int(i), int(j)) for j, i in best.items()), s


def visible_mask(T, PA, RA, shape, s, min_r):
    q = apply_T(T, PA[:, :2]); h, w = shape[:2]; m = 0.05 * min(h, w)
    return (q[:, 0] > -m) & (q[:, 0] < w + m) & (q[:, 1] > -m) & (q[:, 1] < h + m) & (RA * s >= 0.8 * min_r)


def loo_rmse(model, A, B):
    n = len(A)
    if n < MIN_PTS[model] + 1: return float("nan")
    e = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False; T = fit_T(model, A[m], B[m])
        if T is not None and np.all(np.isfinite(T)): e.append(np.linalg.norm(apply_T(T, A[i:i + 1]) - B[i]))
    return float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")


# =============================================================== library seeds (patterns you saved in crater_learn)
def library_seeds(O, ref_path, tgt_path, D, log=print):
    """Every saved pattern is searched in the image it did NOT come from.  A pattern found in both images relates them directly."""
    out = []
    if not O.get("use_library", True): return out
    try: pats = CLN.Store(O.get("out", OUT)).load_patterns()
    except Exception: return out
    if not pats: return out
    ra, rb = os.path.abspath(ref_path), os.path.abspath(tgt_path); low = {}
    for key, img, C in (("A", D["imgA"], D["CA"]), ("B", D["imgB"], D["CB"])):
        low[key] = [(c[0], c[1], math.sqrt(c[2] * c[3])) for c in C if c[5] >= 0.25]
    probes = {}
    for key, img in (("A", D["imgA"]), ("B", D["imgB"])):
        probes[key] = (lambda F, clf: (lambda circ: clf.predict_proba(F(circ.astype(np.float32)))[:, 1]))(CLN.FeatPyr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), D["clf"]) if D.get("clf") is not None else None
    for p in pats:
        pa = os.path.abspath(p.get("image", ""))
        try:
            if pa == ra:                                              # pattern drawn on the reference -> find it in the target
                f = p["scale"] / D["scA"]; q = dict(p, anchor=[v * f for v in p["anchor"]], members=[[v * f for v in m] for m in p["members"]])
                r = CLN.find_pattern(dict(anchor=[v / f * 1 for v in p["anchor"]], members=[[v / f for v in m] for m in p["members"]]), low["B"], probe=probes["B"], frame=D["imgB"].shape[:2])
                if r and r["found"]: out.append((r["T"], f"pattern {p.get('name', '?')} (reference -> target)", r["score"]))
            elif pa == rb:                                            # pattern drawn on the target -> find it in the reference, invert
                f = p["scale"] / D["scB"]; r = CLN.find_pattern(dict(anchor=[v / f for v in p["anchor"]], members=[[v / f for v in m] for m in p["members"]]), low["A"], probe=probes["A"], frame=D["imgA"].shape[:2])
                if r and r["found"]: out.append((np.linalg.inv(r["T"]), f"pattern {p.get('name', '?')} (target -> reference)", r["score"]))
            else:                                                     # pattern from a third image: found in both = direct relation
                ra_ = CLN.find_pattern(p, low["A"], probe=probes["A"], frame=D["imgA"].shape[:2]); rb_ = CLN.find_pattern(p, low["B"], probe=probes["B"], frame=D["imgB"].shape[:2])
                if ra_ and rb_ and ra_["found"] and rb_["found"]: out.append((rb_["T"] @ np.linalg.inv(ra_["T"]), f"pattern {p.get('name', '?')} (found in both)", min(ra_["score"], rb_["score"])))
        except Exception as ex: log(f"  pattern {p.get('name', '?')}: skipped ({type(ex).__name__}: {ex})")
    return out


# =============================================================== 3. hypotheses -> verification -> model -> metrics
def hypothesis_match(PA, PB, shapeA, shapeB, O, seeds=(), log=print):
    """returns dict(T, pairs, model, info) or None.  Hypotheses come from lock-pattern triangles and from saved patterns."""
    RA, RB = radius(PA), radius(PB); treeB = cKDTree(PB[:, :2]); thr = max(5.0, 0.10 * float(np.median(RB))); min_r = float(O["min_r"])
    tA, tB = knn_triangles(PA[:, :2], RA, O["k"]), knn_triangles(PB[:, :2], RB, O["k"]); info = dict(n_tri_ref=0, n_tri_tgt=0, n_close=0, n_seed=len(seeds))
    hyps = [(T, src) for T, src, _ in seeds]
    if tA is not None and tB is not None:
        (da, va), (db, vb) = tA, tB; info.update(n_tri_ref=len(da), n_tri_tgt=len(db)); tree = cKDTree(db); cand = []
        for i, d in enumerate(da):
            for j in tree.query_ball_point(d, max(O["tol"], 0.05) * 1.2, p=np.inf): cand.append((float(np.abs(d - db[j]).max()), i, j))
        cand.sort(); info["n_close"] = len(cand)
        for _, i, j in cand[:7000]:
            T = fit_T("similarity", PA[va[i], :2], PB[vb[j], :2]); s = math.sqrt(abs(np.linalg.det(T[:2, :2])))
            if T is not None and 0.05 < s < 20: hyps.append((T, "triangle"))
    if not hyps: return None
    scored = []
    for T, src in hyps:
        pairs, s = consensus(T, PA, RA, PB, RB, treeB, thr); scored.append((len(pairs), s, T, src, pairs))
        if src == "triangle" and len(pairs) >= 14 and len(pairs) >= 0.85 * min(len(PB), int(visible_mask(T, PA, RA, shapeB, s, min_r).sum())): break
    scored.sort(key=lambda t: -t[0]); n, s, T, src, pairs = scored[0]
    if n < 4: info["best"] = n; return dict(ok=False, info=info)
    second = next((m for m, s2, *_ in scored if abs(math.log(s2 / s)) > 0.25), 0)
    for _ in range(4):                                         # refit on whole-crater centres, re-collect
        if len(pairs) < 3: break
        T2 = fit_T("similarity", PA[[i for i, _ in pairs], :2], PB[[j for _, j in pairs], :2])
        if T2 is None: break
        p2, s2 = consensus(T2, PA, RA, PB, RB, treeB, max(thr * 0.6, 3.0), (0.78, 1.28))
        if len(p2) < 3 or (len(p2) < len(pairs) - 1): break
        T, pairs, s = T2, p2, s2
    info.update(best=len(pairs), scale=s, source=src, second_best_other_zoom=int(second), n_visible_ref=int(visible_mask(T, PA, RA, shapeB, s, min_r).sum()),
                scores=[(a, b) for a, b, *_ in scored[:60]], thr=thr)
    # model choice by leave-one-out error
    A, B = PA[[i for i, _ in pairs], :2], PB[[j for _, j in pairs], :2]; cands = {"similarity": T}; loo = {"similarity": loo_rmse("similarity", A, B)}
    want = ["similarity", "affine", "homography"] if O["model"] == "auto" else [O["model"]]
    for m in want:
        if len(A) >= MIN_PTS[m] + 2 or (O["model"] != "auto" and len(A) >= MIN_PTS[m]):
            Tm = fit_T(m, A, B)
            if Tm is not None and np.all(np.isfinite(Tm)): cands[m] = Tm; loo[m] = loo_rmse(m, A, B)
    if O["model"] == "auto":
        model = "similarity"
        for m in ("affine", "homography"):
            if m in loo and np.isfinite(loo[m]) and np.isfinite(loo[model]) and loo[m] < 0.85 * loo[model] and len(A) >= MIN_PTS[m] + 3: model = m
    else: model = O["model"] if O["model"] in cands else "similarity"
    info["loo_by_model"] = {k: (None if not np.isfinite(v) else round(float(v), 3)) for k, v in loo.items()}
    return dict(ok=True, T=cands[model], T_sim=T, pairs=pairs, model=model, info=info, PA=PA, PB=PB, thr=thr)


# =============================================================== 4. optional dense sub-pixel refinement (ECC) - accepted only if the craters agree at least as well
def ecc_refine(D, T, pairs, PA, PB, model, log=print):
    try:
        fA, fB = CLN.Feat(cv2.cvtColor(D["imgA"], cv2.COLOR_BGR2GRAY)), CLN.Feat(cv2.cvtColor(D["imgB"], cv2.COLOR_BGR2GRAY))
        a = np.clip(fA.mag / fA.g95 * 80, 0, 255).astype(np.float32); b = np.clip(fB.mag / fB.g95 * 80, 0, 255).astype(np.float32)
        h, w = a.shape; valid = cv2.warpPerspective(np.full(b.shape, 255, np.uint8), T, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST); valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        mt = cv2.MOTION_HOMOGRAPHY if model == "homography" else cv2.MOTION_AFFINE; W = T.astype(np.float32).copy() if mt == cv2.MOTION_HOMOGRAPHY else T[:2].astype(np.float32).copy()
        cc, W = cv2.findTransformECC(a, b, W, mt, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6), valid, 5)
        T2 = W.astype(np.float64) if mt == cv2.MOTION_HOMOGRAPHY else np.vstack([W.astype(np.float64), [0, 0, 1]])
        A, B = PA[[i for i, _ in pairs], :2], PB[[j for _, j in pairs], :2]; e0 = float(np.sqrt(np.mean(np.sum((apply_T(T, A) - B) ** 2, 1)))); e1 = float(np.sqrt(np.mean(np.sum((apply_T(T2, A) - B) ** 2, 1))))
        ok = np.all(np.isfinite(T2)) and e1 <= 1.1 * e0 + 0.3 and cc > 0.15
        log(f"  ECC: correlation {cc:.3f}, crater residual {e0:.2f} -> {e1:.2f} px : {'accepted' if ok else 'rejected (kept the crater-based fit)'}")
        return (T2 if ok else T), dict(used=bool(ok), cc=float(cc), before=e0, after=e1)
    except Exception as ex:
        log(f"  ECC skipped ({type(ex).__name__})"); return T, dict(used=False, error=str(ex))


def metrics_of(T, M, D, O, geoA=None):
    PA, PB, pairs = M["PA"], M["PB"], M["pairs"]; A, B = PA[[i for i, _ in pairs], :2], PB[[j for _, j in pairs], :2]
    hA, wA = D["imgA"].shape[:2]; hB, wB = D["imgB"].shape[:2]; s, th = sim_params(T[:2, :2] if False else np.vstack([T[:2], [0, 0, 1]]))
    s = math.sqrt(abs(np.linalg.det(T[:2, :2]))); resB = np.linalg.norm(apply_T(T, A) - B, axis=1); resA = resB / max(s, 1e-9)
    Ti = np.linalg.inv(T); RA, RB = radius(PA), radius(PB)
    visA = int(visible_mask(T, PA, RA, (hB, wB), s, O["min_r"]).sum()); visB = int(visible_mask(Ti, PB, RB, (hA, wA), 1 / s, O["min_r"]).sum())
    corners = apply_T(Ti, np.array([[0, 0], [wB, 0], [wB, hB], [0, hB]], float)).astype(np.float32); frame = np.array([[0, 0], [wA, 0], [wA, hA], [0, hA]], np.float32)
    try: ok, poly = cv2.intersectConvexConvex(corners, frame); poly = poly.reshape(-1, 2) if ok > 0 else frame
    except Exception: poly = frame
    area = float(abs(cv2.contourArea(poly.astype(np.float32)))) or 1.0
    Ain = A; hull = cv2.convexHull(Ain.astype(np.float32)) if len(Ain) >= 3 else None; hull_frac = float(cv2.contourArea(hull) / area) if hull is not None else 0.0
    x0, y0 = poly.min(0); x1, y1 = poly.max(0); cells = hit = 0
    for i in range(3):
        for j in range(3):
            c = ((x0 + (i + .5) * (x1 - x0) / 3), (y0 + (j + .5) * (y1 - y0) / 3))
            if cv2.pointPolygonTest(poly.astype(np.float32), c, False) < 0: continue
            cells += 1; hit += int(((A[:, 0] >= x0 + i * (x1 - x0) / 3) & (A[:, 0] < x0 + (i + 1) * (x1 - x0) / 3) & (A[:, 1] >= y0 + j * (y1 - y0) / 3) & (A[:, 1] < y0 + (j + 1) * (y1 - y0) / 3)).any())
    loo = loo_rmse(M["model"], A, B); loo = float("nan") if not np.isfinite(loo) else loo
    m = dict(model=M["model"], n_craters_ref=int(len(D["CA"])), n_craters_tgt=int(len(D["CB"])), n_pattern_ref=int(len(PA)), n_pattern_tgt=int(len(PB)), n_inliers=int(len(pairs)),
             n_visible_ref=visA, n_visible_tgt=visB, inlier_ratio=round(len(pairs) / max(1, min(visA, visB)), 3), zoom_tgt_over_ref=round(s, 5), rotation_deg=round(math.degrees(math.atan2(T[1, 0], T[0, 0])), 3),
             translation_px=[round(float(T[0, 2]), 2), round(float(T[1, 2]), 2)], rmse_ref_px=round(float(np.sqrt((resA ** 2).mean())), 3), rmse_tgt_px=round(float(np.sqrt((resB ** 2).mean())), 3),
             median_err_ref_px=round(float(np.median(resA)), 3), max_err_ref_px=round(float(resA.max()), 3), heldout_rmse_tgt_px=None if np.isnan(loo) else round(loo, 3),
             heldout_rmse_ref_px=None if np.isnan(loo) else round(loo / max(s, 1e-9), 3), coverage_3x3=round(hit / max(cells, 1), 3), hull_fraction_of_overlap=round(hull_frac, 3),
             overlap_fraction_of_ref=round(area / (wA * hA), 3), perspective_terms=[round(float(T[2, 0]), 8), round(float(T[2, 1]), 8)] if M["model"] == "homography" else None)
    inf = M["info"]; m["uniqueness"] = round(inf["best"] / max(inf["best"], inf.get("second_best_other_zoom", 0), 1), 3); m["hypothesis_source"] = inf.get("source")
    m["subpixel_note"] = "centres are ellipse-refined to a fraction of a pixel; errors are in WORK pixels (image shrunk to max_side)"
    if geoA is not None: m["rmse_metres"] = round(m["rmse_ref_px"] * geoA.gsd_work(D["imgA"].shape), 2); m["ref_work_px_metres"] = round(geoA.gsd_work(D["imgA"].shape), 3)
    return m, resA, poly


# =============================================================== lock pattern picture (same star in both images)
def lock_pattern_panel(imgA, PA, imgB, PB, pairs, k_show=7):
    ia, ib = np.array([i for i, _ in pairs]), np.array([j for _, j in pairs])
    if len(ia) < 3: return None, float("nan")
    best, score = 0, -1
    for n in range(len(ia)):
        d = np.hypot(*(PA[ia, :2] - PA[ia[n], :2]).T); sc = (d < 4.0 * np.sort(d)[min(3, len(d) - 1)] + 1e-9).sum() + 0.02 * radius(PA[ia[n:n + 1]])[0]
        if sc > score: best, score = n, sc
    a0 = best; dA = np.hypot(*(PA[ia, :2] - PA[ia[a0], :2]).T); order = [n for n in np.argsort(dA) if n != a0][:k_show]
    cols = [tuple(int(x) for x in cv2.cvtColor(np.uint8([[[(n * 40) % 180, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for n in range(len(order))]; vA, vB = imgA.copy(), imgB.copy()
    for P_, idx, v in ((PA, ia, vA), (PB, ib, vB)):
        c0 = (int(P_[idx[a0], 0]), int(P_[idx[a0], 1]))
        for col, n in zip(cols, order):
            cv2.line(v, c0, (int(P_[idx[n], 0]), int(P_[idx[n], 1])), col, 2, cv2.LINE_AA); draw_ell(v, P_[idx[n]], col, 2)
        draw_ell(v, P_[idx[a0]], (0, 0, 255), 3)
    S = max(360, min(imgA.shape[0], imgB.shape[0])); can = np.full((S, S, 3), 25, np.uint8); tag(can, "normalised star: A = filled, B = ring", 18); ref = order[0]
    rrmax = max(np.hypot(*(PA[ia[n], :2] - PA[ia[a0], :2])) / np.hypot(*(PA[ia[ref], :2] - PA[ia[a0], :2])) for n in order); UNIT = (S / 2 - 26) / max(rrmax, 1.0); pts = {}
    for key, (P_, idx, filled) in {"A": (PA, ia, True), "B": (PB, ib, False)}.items():
        c0 = P_[idx[a0], :2]; v0 = P_[idx[ref], :2] - c0; L0, ang0 = np.hypot(*v0), math.atan2(v0[1], v0[0])
        for col, n in zip(cols, order):
            v = P_[idx[n], :2] - c0; rr, an = np.hypot(*v) / L0, math.atan2(v[1], v[0]) - ang0 - math.pi / 2; p = (int(S / 2 + UNIT * rr * math.cos(an)), int(S / 2 + UNIT * rr * math.sin(an))); pts.setdefault(n, {})[key] = p
            cv2.line(can, (S // 2, S // 2), p, col, 1, cv2.LINE_AA); cv2.circle(can, p, max(3, int(UNIT * radius(P_[idx[n]:idx[n] + 1])[0] / L0)), col, -1 if filled else 2, cv2.LINE_AA)
    cv2.circle(can, (S // 2, S // 2), 6, (0, 0, 255), -1); dev = [math.hypot(v["A"][0] - v["B"][0], v["A"][1] - v["B"][1]) / UNIT for v in pts.values() if len(v) == 2]
    return hstack_pad(hstack_pad(vA, vB), can), (float(np.mean(dev)) if dev else float("nan"))


# =============================================================== baseline: plain SIFT + RANSAC (what most solutions do) for comparison
def baseline_sift(D, T):
    ga, gb = cv2.cvtColor(D["imgA"], cv2.COLOR_BGR2GRAY), cv2.cvtColor(D["imgB"], cv2.COLOR_BGR2GRAY); sift = cv2.SIFT_create(4000, contrastThreshold=0.01)
    ka, da = sift.detectAndCompute(ga, None); kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return dict(matches=0, inliers=0, inlier_ratio=0.0, note="too few features")
    mm = cv2.BFMatcher().knnMatch(da, db, k=2); good = [a for a, b in (m for m in mm if len(m) == 2) if a.distance < 0.8 * b.distance]
    if len(good) < 8: return dict(matches=len(good), inliers=0, inlier_ratio=0.0, note="too few matches")
    pa = np.float32([ka[m.queryIdx].pt for m in good]); pb = np.float32([kb[m.trainIdx].pt for m in good]); H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    if H is None: return dict(matches=len(good), inliers=0, inlier_ratio=0.0, note="RANSAC failed")
    n = int(inl.sum()); h, w = ga.shape; g = np.array([[x, y] for x in np.linspace(w * .15, w * .85, 5) for y in np.linspace(h * .15, h * .85, 5)])
    ag = float(np.median(np.linalg.norm(apply_T(H, g) - apply_T(T, g), axis=1)))
    return dict(matches=len(good), inliers=n, inlier_ratio=round(n / len(good), 3), median_disagreement_with_ours_px=round(ag, 2), agrees=bool(ag < 6 and n >= 12))


# =============================================================== accuracy: how well do the images really overlap (sub-pixel check)
def dense_check(imgA, reg, valid, n=10, hw=20, sw=14, min_score=0.3):
    """Residual misregistration measured on the IMAGES, not on the craters: patches of the registered target are correlated against the
    reference (locally normalised + gradient images = lighting-tolerant), the correlation peak is refined with a parabola.  The shift of
    that peak is the remaining error at that spot.  Independent of how well the crater centres were detected."""
    ga, gr = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY); FA, FR = CLN.Feat(ga), CLN.Feat(gr)
    var = [(FA.ln.astype(np.float32), FR.ln.astype(np.float32)), (np.clip(FA.mag / FA.g95 * 60, 0, 255).astype(np.float32), np.clip(FR.mag / FR.g95 * 60, 0, 255).astype(np.float32))]
    h, w = ga.shape; k = 2 * (hw + sw) + 1; er = cv2.erode(valid.astype(np.uint8), np.ones((k, k), np.uint8)); nodes = []; tried = 0; rejected = 0
    if h - hw - sw - 3 <= hw + sw + 2 or w - hw - sw - 3 <= hw + sw + 2: return dict(n=0, tried=0, rejected=0, nodes=np.zeros((0, 5)))
    for y in np.linspace(hw + sw + 2, h - hw - sw - 3, n):
        for x in np.linspace(hw + sw + 2, w - hw - sw - 3, n):
            xi, yi = int(round(x)), int(round(y))
            if not er[yi, xi]: continue
            tried += 1; best = None
            for A_, R_ in var:
                T_ = R_[yi - hw:yi + hw + 1, xi - hw:xi + hw + 1]
                if T_.std() < 0.1 * (A_.std() + 1e-6): continue
                S_ = A_[yi - hw - sw:yi + hw + sw + 1, xi - hw - sw:xi + hw + sw + 1]; res = cv2.matchTemplate(S_, T_, cv2.TM_CCOEFF_NORMED); _, mx, _, (lx, ly) = cv2.minMaxLoc(res); dx = dy = 0.0
                if 0 < lx < res.shape[1] - 1:
                    a, b, c = res[ly, lx - 1], res[ly, lx], res[ly, lx + 1]; dx = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
                if 0 < ly < res.shape[0] - 1:
                    a, b, c = res[ly - 1, lx], res[ly, lx], res[ly + 1, lx]; dy = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
                if best is None or mx > best[0]: best = (mx, lx + dx - sw, ly + dy - sw)
            if best is not None and best[0] >= min_score:
                if max(abs(best[1]), abs(best[2])) > sw - 1.5: rejected += 1; continue                      # peak at the edge of the search window = false match, not a measurement
                nodes.append((xi, yi, best[1], best[2], best[0]))
    nd = np.array(nodes, float).reshape(-1, 5)
    if len(nd) == 0: return dict(n=0, tried=tried, rejected=rejected, nodes=nd)
    mag = np.hypot(nd[:, 2], nd[:, 3]); med = float(np.median(mag)); mad = 1.4826 * float(np.median(np.abs(mag - med))); keep = mag <= max(1.5, med + 4 * max(mad, 0.2))
    return dict(n=int(len(nd)), tried=tried, rejected=rejected, trimmed=int((~keep).sum()), rms=float(np.sqrt((mag[keep] ** 2).mean())), rms_all=float(np.sqrt((mag ** 2).mean())), median=med, within_05=float((mag <= 0.5).mean()),
                within_1=float((mag <= 1.0).mean()), max=float(mag[keep].max()), mean_dx=float(nd[:, 2].mean()), mean_dy=float(nd[:, 3].mean()), nodes=nd)


def grade_of(px):
    if px is None or not np.isfinite(px): return "unknown", "#999999"
    if px <= 0.5: return "EXCELLENT - sub-pixel (<= 0.5 px)", "#2ecc71"
    if px <= 1.0: return "SUB-PIXEL (<= 1 px)", "#7bd96b"
    if px <= 2.0: return "near sub-pixel (1 - 2 px)", "#f1c40f"
    if px <= 5.0: return "pixel-level (2 - 5 px)", "#f39c12"
    return "COARSE (> 5 px)", "#e74c3c"


def quality_fields(m, dc, D, T, ecc_used=False):
    s, scB = m["zoom_tgt_over_ref"], D["scB"]
    f = dict(rmse_src_orig_px=round(m["rmse_tgt_px"] / scB, 3), heldout_rmse_src_orig_px=None if m["heldout_rmse_tgt_px"] is None else round(m["heldout_rmse_tgt_px"] / scB, 3))
    use_dense = dc["n"] >= 8
    if dc["n"] > 0:
        f.update(dense_nodes=dc["n"], dense_nodes_tried=dc["tried"], dense_rejected=dc.get("rejected", 0), dense_trimmed=dc.get("trimmed", 0), dense_rms_ref_px=round(dc["rms"], 3), dense_median_ref_px=round(dc["median"], 3), dense_within_0p5px=round(dc["within_05"], 3), dense_within_1px=round(dc["within_1"], 3),
                 dense_rms_src_orig_px=round(dc["rms"] * s / scB, 3))
        if "ref_work_px_metres" in m: f["dense_rms_metres"] = round(dc["rms"] * m["ref_work_px_metres"], 2)
    basis = f["dense_rms_src_orig_px"] if use_dense else (f["heldout_rmse_src_orig_px"] if f["heldout_rmse_src_orig_px"] is not None else f["rmse_src_orig_px"])
    grade, color = grade_of(basis)
    f.update(accuracy_src_px=round(float(basis), 3), accuracy_basis="residual shift measured on the images (lighting-tolerant correlation)" if use_dense else "held-out crater-centre error (too few image patches matched)", quality=grade, quality_color=color)
    dpart = f" | image residual {f['dense_rms_src_orig_px']} px" if dc["n"] > 0 else ""
    f["quality_banner"] = f"RMSE {f['rmse_src_orig_px']} px (craters, held-out {f['heldout_rmse_src_orig_px']}){dpart} | {grade}   [pixels of the source image]"
    f["quality_text"] = (f"Accuracy {grade}: {f['accuracy_src_px']} px in the SOURCE image's own pixels, based on {f['accuracy_basis']}. Crater RMSE {f['rmse_src_orig_px']} px "
                         f"(held-out {f['heldout_rmse_src_orig_px']}). " + (f"{dc['n']} of {dc['tried']} image patches matched; {dc['within_1']:.0%} are within 1 px, {dc['within_05']:.0%} within 0.5 px, worst kept {dc['max']:.2f} px (reference work pixels); {dc.get('rejected', 0)} edge-of-window false matches and {dc.get('trimmed', 0)} extreme outliers were excluded from the RMS. " if dc["n"] > 0 else "")
                         + "Crater-centre error includes detection noise, so it is a conservative figure; the image residual measures what is left after alignment."
                         + (" NOTE: dense refinement (ECC) was used and optimises a similar image correlation, so the image residual can be optimistic - the held-out crater error is the independent cross-check." if ecc_used else ""))
    return f


def dense_map_image(imgA, dc, m):
    v = cv2.cvtColor(cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR); v = (v * 0.7).astype(np.uint8)
    for x, y, dx, dy, sc in dc["nodes"]:
        mag = math.hypot(dx, dy); col = (80, 220, 80) if mag <= 0.5 else ((60, 220, 240) if mag <= 1.0 else (60, 60, 240)); cv2.circle(v, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
        cv2.arrowedLine(v, (int(x), int(y)), (int(x + 10 * dx), int(y + 10 * dy)), col, 2, cv2.LINE_AA, tipLength=0.3)
    tag(v, "residual shift per patch (arrows x10): green <=0.5 px, yellow <=1 px, red >1 px", 22); tag(v, m.get("quality_banner", ""), v.shape[0] - 12); return v


# =============================================================== orchestrator
def register_pair(ref_path, tgt_path, O=None, log=print):
    O = {**DEFAULTS, **(O or {})}; t0 = time.time(); steps = []; R = dict(ok=False, steps=steps, options={k: v for k, v in O.items() if not callable(v)}, ref=str(ref_path), tgt=str(tgt_path), notes=[])
    R["xml_ref"], R["xml_tgt"] = parse_xml(O.get("ref_xml")), parse_xml(O.get("tgt_xml"))
    log("1/6 craters ..."); D = detect_pair(ref_path, tgt_path, O, log); R["D"] = D; R["notes"] += D["notes"]; imgA, imgB = D["imgA"], D["imgB"]
    geoA = CsvGeo(O["ref_csv"], D["oA"], D["scA"]) if O.get("ref_csv") else None; geoB = CsvGeo(O["tgt_csv"], D["oB"], D["scB"]) if O.get("tgt_csv") else None
    steps.append(("1  Reference: detected craters", tag(draw_set(imgA, D["CA"], (0, 255, 0)), f"reference: {len(D['CA'])} craters"), "Green = reference craters (ellipses refined to sub-pixel)."))
    steps.append(("2  Target: detected craters", tag(draw_set(imgB, D["CB"], (255, 0, 255)), f"target: {len(D['CB'])} craters"), "Magenta = target craters."))
    log("2/6 saved patterns ..."); seeds = library_seeds(O, ref_path, tgt_path, D, log); R["seeds"] = [(s[1], round(float(s[2]), 3)) for s in seeds]
    if seeds: log(f"  {len(seeds)} saved pattern(s) recognised in the pair")
    log("3/6 pattern matching ..."); M = None
    for p in dict.fromkeys([float(O["ml_p"]), 0.4, 0.3, 0.25]):                     # auto-sensitivity: accept less sure craters until a pattern is found
        CA = D["CA"][D["CA"][:, 5] >= p][:int(O["max_craters"])]; CB = D["CB"][D["CB"][:, 5] >= p][:int(O["max_craters"])]
        ia, ib = select_pattern(CA, imgA.shape, O["min_r"], O["max_frac"], O["max_pattern"]), select_pattern(CB, imgB.shape, O["min_r"], O["max_frac"], O["max_pattern"])
        if len(ia) < 4 or len(ib) < 4: log(f"  confidence >= {p}: only {len(ia)} / {len(ib)} usable craters"); continue
        M = hypothesis_match(CA[ia], CB[ib], imgA.shape, imgB.shape, O, seeds, log)
        if M and M.get("ok"): R["ml_conf_used"] = p; D["CA_used"], D["CB_used"] = CA, CB; break
        log(f"  no pattern at confidence >= {p}")
    pa_img = hstack_pad(tag(draw_set(imgA, D["CA"][select_pattern(D["CA"], imgA.shape, O["min_r"], O["max_frac"], O["max_pattern"])], (0, 255, 0)), "pattern craters: reference"), tag(draw_set(imgB, D["CB"][select_pattern(D["CB"], imgB.shape, O["min_r"], O["max_frac"], O["max_pattern"])], (255, 0, 255)), "pattern craters: target"))
    steps.append(("3  Pattern craters (medium, not cut by the border)", pa_img, "Only craters of useful size that the border does not cut are used for the pattern."))
    if not M or not M.get("ok"):
        R["message"] = "No consistent crater pattern: fewer than 4 craters line up under any zoom/rotation. Detect more craters (train the model in crater_learn, lower 'Min radius'), save patterns, or check that the images overlap."
        steps.append(("4  Matching", pa_img, R["message"])); R["metrics"] = dict(n_inliers=0); R["seconds"] = round(time.time() - t0, 1); return R
    inf = M["info"]; hg = np.full((300, 700, 3), 25, np.uint8); sc = M["info"]["scores"][:40]; hmax = max(sc[0][0], 1)
    for i, (n, sz) in enumerate(sc): cv2.rectangle(hg, (int(30 + 640 * i / len(sc)), 270 - int(220 * n / hmax)), (int(30 + 640 * (i + 1) / len(sc)) - 2, 270), (0, 200, 0) if i == 0 else (120, 120, 120), -1)
    tag(hg, "craters that line up for the 40 best hypotheses (green = chosen)"); cv2.putText(hg, f"zoom reference->target = x{inf['scale']:.3f}", (30, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    steps.append(("4  Pattern votes: each look-alike triangle proposes a zoom / rotation / shift", hg, f"{inf['n_tri_ref']} triangles in the reference, {inf['n_tri_tgt']} in the target, {inf['n_close']} look alike; {inf['n_seed']} came from patterns you saved. "
                  f"Chosen: zoom x{inf['scale']:.3f}, {inf['best']} craters line up ({inf['n_visible_ref']} reference craters should be visible). Best competing different zoom: {inf['second_best_other_zoom']}. Source of the winning hypothesis: {inf['source']}."))
    T = M["T"]; log(f"4/6 model: {M['model']}  (leave-one-out error by model: {inf['loo_by_model']})"); ecc = None
    if O.get("use_ecc"): log("  dense sub-pixel refinement (ECC) ..."); T, ecc = ecc_refine(D, T, M["pairs"], M["PA"], M["PB"], M["model"], log)
    R["ecc"] = ecc; M["T_final"] = T; geo_for_m = geoA
    m, resA, poly = metrics_of(T, M, D, O, geoA); R["metrics"] = m; R["T_ref_to_tgt"] = T.tolist(); R["M"] = M; R["ok"] = True
    pairs = M["pairs"]; big = hstack_pad(imgA.copy(), imgB.copy()); off = imgA.shape[1] + 8
    for k, (i, j) in enumerate(pairs):
        col = tuple(int(x) for x in cv2.cvtColor(np.uint8([[[(k * 23) % 180, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]); draw_ell(big, M["PA"][i], col, 2); b = M["PB"][j].copy(); b[0] += off; draw_ell(big, b, col, 2)
        cv2.line(big, (int(M["PA"][i, 0]), int(M["PA"][i, 1])), (int(b[0]), int(b[1])), col, 1, cv2.LINE_AA)
    steps.append(("5  Matched craters (same colour = same crater)", tag(big, f"{len(pairs)} craters matched"), f"Position AND size must agree. Model: {M['model']}; held-out RMSE {m['heldout_rmse_ref_px']} px (reference pixels)."))
    lp, dev = lock_pattern_panel(imgA, M["PA"], imgB, M["PB"], pairs)
    if lp is not None: steps.append(("6  LOCK PATTERN: the same star in both images", lp, f"Red = anchor crater; lines go from its centre to its neighbours. Right: both stars normalised (anchor in the middle, nearest neighbour 'up' = 1). Filled = reference, ring = target. Mean difference {dev:.3f}."))
    log("5/6 registered product ..."); h, w = imgA.shape[:2]
    reg = cv2.warpPerspective(imgB, T, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR); valid = cv2.warpPerspective(np.full(imgB.shape[:2], 255, np.uint8), T, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST) > 0
    ga = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY); gr = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY); ov = np.dstack([ga, ga, np.where(valid, gr, ga)])
    yy, xx = np.mgrid[0:h, 0:w]; chk = ((yy // 48 + xx // 48) % 2).astype(bool); cb = np.where((chk & valid)[..., None], reg, imgA)
    dif = np.abs(CLN.local_norm(ga.astype(np.float32)) - CLN.local_norm(gr.astype(np.float32))); diff = cv2.applyColorMap(np.where(valid, np.clip(dif / 3 * 255, 0, 255), 0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    R["products"] = dict(registered=np.where(valid[..., None], reg, 0).astype(np.uint8), overlay=ov, checker=cb, diff=diff, matches=big, lock=lp, valid=valid)
    log("  accuracy check on the images (sub-pixel) ..."); dc = dense_check(imgA, reg, valid); R["dense"] = {k: v for k, v in dc.items() if k != "nodes"}; m.update(quality_fields(m, dc, D, T, bool(R.get("ecc") and R["ecc"].get("used")))); R["products"]["accuracy"] = dense_map_image(imgA, dc, m)
    steps.append(("7  Overlay (target in red channel)", ov, "Where the images agree the colours merge to grey; fringes show residual misalignment or different lighting."))
    steps.append(("8  Checkerboard", cb, "Alternating tiles of reference and registered target: features must continue across tile borders."))
    sT = sim_params(np.vstack([T[:2], [0, 0, 1]]))[0] if False else math.sqrt(abs(np.linalg.det(T[:2, :2]))); rotT = math.degrees(math.atan2(T[1, 0], T[0, 0])); Ti = np.linalg.inv(T); mapped = []
    for _, j in pairs:
        c = M["PB"][j]; mapped.append([*apply_T(Ti, c[None, :2])[0], c[2] / sT, c[3] / sT, c[4] - rotT])
    both = hstack_pad(draw_set(imgA, M["PA"][[i for i, _ in pairs]], (0, 255, 0)), draw_set(imgA, np.array(mapped).reshape(-1, 5), (255, 0, 255))) if pairs else None
    if both is not None: steps.append(("9  Crater outlines after registration", tag(both, "left: reference craters | right: target craters mapped into the reference frame"), "Matched craters should sit on the same craters of the reference."))
    steps.append(("10  ACCURACY: how well do the images overlap (sub-pixel check)", R["products"]["accuracy"], m["quality_text"]))
    if O.get("baseline"): R["baseline"] = baseline_sift(D, T); log(f"  baseline SIFT+RANSAC: {R['baseline']}")
    # ----- geo: lon/lat of the matches + a grid CSV for the target
    if geoA is not None:
        ax = M["PA"][[i for i, _ in pairs], :2]; lon, lat = geoA.work2ll(ax[:, 0], ax[:, 1]); R["match_lonlat"] = (lon, lat)
        Wt, Ht = D["oB"]; Wf, Hf = (geoB.full_size if geoB is not None else (Wt, Ht)); px = np.arange(0, Wf, 100); sc_ = np.arange(0, Hf, 100)
        px = px if px[-1] == Wf - 1 else np.append(px, Wf - 1); sc_ = sc_ if sc_[-1] == Hf - 1 else np.append(sc_, Hf - 1); PX, SC = np.meshgrid(px, sc_); PX, SC = PX.ravel(), SC.ravel()
        tw = np.c_[((PX + .5) * Wt / Wf - .5 + .5) * D["scB"] - .5, ((SC + .5) * Ht / Hf - .5 + .5) * D["scB"] - .5]; rw = apply_T(np.linalg.inv(T), tw); glon, glat = geoA.work2ll(rw[:, 0], rw[:, 1])
        R["grid"] = dict(Longitude=glon, Latitude=glat, Pixel=PX, Scan=SC)
        if geoB is not None:
            tl, tb = geoB.work2ll(tw[:, 0], tw[:, 1]); g = np.isfinite(tl) & np.isfinite(glon); p1, p2, dl = np.radians(tb[g]), np.radians(glat[g]), np.radians(tl[g] - glon[g])
            d = R_MOON * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1)); R["vs_official_csv_m"] = dict(mean=round(float(d.mean()), 2), median=round(float(np.median(d)), 2), max=round(float(d.max()), 2), n=int(g.sum()))
    # ----- xml-derived context
    fr, ft = R["xml_ref"]["facts"], R["xml_tgt"]["facts"]; ctx = {}
    ta, tb_ = _t(fr.get("start_time", "")), _t(ft.get("start_time", ""))
    if ta and tb_: ctx["acquisition_time_difference_days"] = round(abs((tb_ - ta).total_seconds()) / 86400, 4)
    for k in ("sun_elevation_deg", "sun_azimuth_deg", "incidence_angle_deg"):
        if k in fr and k in ft: ctx["delta_" + k] = round(abs(ft[k] - fr[k]), 2)
    R["xml_context"] = ctx
    if O.get("stress"): log("6/6 stress test with known ground truth ..."); R["stress"] = stress_test(ref_path, O, log)
    R["seconds"] = round(time.time() - t0, 1); log(f"done in {R['seconds']} s | {len(pairs)} craters matched | model {M['model']} | RMSE {m['rmse_ref_px']} px (held-out {m['heldout_rmse_ref_px']}) | zoom x{m['zoom_tgt_over_ref']} rot {m['rotation_deg']} deg | ACCURACY: {m['quality']} ({m['accuracy_src_px']} px)")
    return R


# =============================================================== stress test: register the reference against copies of itself with KNOWN geometry (ground truth) and changed lighting
def stress_test(ref_path, O, log=print):
    ref = cv2.imread(ref_path); h0, w0 = ref.shape[:2]; k = min(1.0, O["max_side"] / max(h0, w0)); ref = cv2.resize(ref, None, fx=k, fy=k, interpolation=cv2.INTER_AREA); h, w = ref.shape[:2]
    cases = [("rotate 25, zoom 0.85, brighter/ramp", 25, 0.85, 1.4, 0.5), ("rotate -50, zoom 1.6 (zoomed in), contrast", -50, 1.6, 0.7, 0.4), ("rotate 80, zoom 0.55 (zoomed out), gamma", 80, 0.55, 1.8, 0.6)]; rows = []
    import tempfile
    d = tempfile.mkdtemp(); rp = os.path.join(d, "ref.png"); cv2.imwrite(rp, ref); rng = np.random.RandomState(0)
    for name, rot, zm, gam, ramp in cases:
        M2 = cv2.getRotationMatrix2D((w / 2, h / 2), rot, zm); A = np.vstack([M2, [0, 0, 1]]); src = cv2.warpAffine(ref, M2, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); src = np.clip(src * (1 + ramp * (xx / w - 0.5))[..., None], 0, 1) ** gam; src = cv2.GaussianBlur(src, (0, 0), 0.8) + rng.randn(h, w, 3).astype(np.float32) * 0.015
        tp = os.path.join(d, "tgt.png"); cv2.imwrite(tp, np.clip(src * 255, 0, 255).astype(np.uint8))
        try:
            O2 = {**O, "stress": False, "baseline": False, "use_ecc": False, "make_report": False, "ref_xml": None, "tgt_xml": None, "ref_csv": None, "tgt_csv": None, "ref_craters": None, "tgt_craters": None, "use_library": False}
            r = register_pair(rp, tp, O2, log=lambda s: None)
            if r["ok"]:
                g = np.array([[x, y_] for x in np.linspace(w * .05, w * .95, 12) for y_ in np.linspace(h * .05, h * .95, 12)]); q = apply_T(A, g); g = g[(q[:, 0] > 0.05 * w) & (q[:, 0] < 0.95 * w) & (q[:, 1] > 0.05 * h) & (q[:, 1] < 0.95 * h)]      # only where the target really looks
                e = np.linalg.norm(apply_T(np.array(r["T_ref_to_tgt"]), g) - apply_T(A, g), axis=1)
                rows.append(dict(case=name, ok=True, craters_matched=r["metrics"]["n_inliers"], rmse_vs_truth_px=round(float(np.sqrt((e ** 2).mean())), 2), zoom_found=r["metrics"]["zoom_tgt_over_ref"], zoom_true=zm))
            else: rows.append(dict(case=name, ok=False))
        except Exception as ex: rows.append(dict(case=name, ok=False, error=str(ex)[:80]))
        log(f"  stress '{name}': {rows[-1]}")
    return rows


# =============================================================== saving + report
def _img_b64(img, maxw=1000, q=82):
    if img is None: return ""
    k = min(1.0, maxw / img.shape[1]); im = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q]); return base64.b64encode(buf).decode()


def _kv(d): return "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(json.dumps(v) if isinstance(v, (list, dict)) else str(v))}</td></tr>" for k, v in d.items())


def build_report(R, out_dir):
    m = R["metrics"]; xr, xt = R["xml_ref"], R["xml_tgt"]; P = R.get("products", {}); css = ("body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;margin:24px auto;color:#222}h1{color:#0b3d91}h2{border-bottom:2px solid #0b3d91;padding-bottom:3px;margin-top:28px}"
           "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px;text-align:left}th{background:#eef}.ok{color:#080;font-weight:bold}.bad{color:#b00;font-weight:bold}img{max-width:100%;border:1px solid #999}.small{font-size:12px;color:#555}")
    h = [f"<html><head><meta charset='utf-8'><title>Registration report</title><style>{css}</style></head><body>", "<h1>Crater-pattern registration report</h1>",
         f"<p class='small'>generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; reference: <b>{html.escape(os.path.basename(R['ref']))}</b> &nbsp;|&nbsp; source/target: <b>{html.escape(os.path.basename(R['tgt']))}</b> &nbsp;|&nbsp; run time {R.get('seconds')} s</p>"]
    h.append(f"<h2>1. Result</h2><p class='{'ok' if R['ok'] else 'bad'}'>{'REGISTERED' if R['ok'] else 'NOT REGISTERED'}</p>")
    if not R["ok"]: h.append(f"<p>{html.escape(R.get('message', ''))}</p></body></html>"); return "".join(h)
    h.append("<table><tr><th>metric</th><th>value</th><th>meaning</th></tr>")
    for k, v, d in (("transform model", m["model"], "chosen automatically by leave-one-out error"), ("matched craters (inliers)", m["n_inliers"], "craters whose position AND size agree"),
                    ("inlier ratio", m["inlier_ratio"], "inliers / craters that should be visible in both images"), ("zoom (target / reference)", m["zoom_tgt_over_ref"], "scale ratio found"),
                    ("rotation (deg)", m["rotation_deg"], ""), ("RMSE (reference px)", m["rmse_ref_px"], "residual of the matched crater centres"),
                    ("held-out RMSE (reference px)", m["heldout_rmse_ref_px"], "leave-one-out: each crater predicted by a fit that never saw it"), ("median / max error (ref px)", f"{m['median_err_ref_px']} / {m['max_err_ref_px']}", ""),
                    ("coverage 3x3", m["coverage_3x3"], "fraction of overlap cells containing a match (uniform distribution)"), ("hull / overlap area", m["hull_fraction_of_overlap"], "how much of the overlap the matches span"),
                    ("hypothesis uniqueness", m["uniqueness"], "best zoom vs best competing zoom (1.0 = unambiguous)")):
        h.append(f"<tr><td>{k}</td><td><b>{v}</b></td><td class='small'>{d}</td></tr>")
    if "rmse_metres" in m: h.append(f"<tr><td>RMSE in metres</td><td><b>{m['rmse_metres']}</b></td><td class='small'>using the reference grid CSV (1 work px = {m['ref_work_px_metres']} m)</td></tr>")
    h.append("</table>")
    if R.get("vs_official_csv_m"): h.append(f"<p>Generated target grid compared with the target's OFFICIAL grid CSV: {html.escape(json.dumps(R['vs_official_csv_m']))} (metres).</p>")
    if R.get("ecc"): h.append(f"<p class='small'>Dense refinement (ECC): {html.escape(json.dumps(R['ecc']))}</p>")
    h.append(f"<h2>1b. Accuracy: how well do the images overlap?</h2><p><span style='font-size:20px;font-weight:bold;color:{m.get('quality_color', '#333')}'>{html.escape(m.get('quality', ''))}</span> &nbsp; ({m.get('accuracy_src_px')} px in the source image's own pixels)</p><p>{html.escape(m.get('quality_text', ''))}</p>")
    h.append("<table><tr><th>measure</th><th>work px</th><th>source-image px</th><th>what it tells</th></tr>")
    h.append(f"<tr><td>crater-centre RMSE</td><td>{m['rmse_ref_px']} (ref)</td><td>{m.get('rmse_src_orig_px')}</td><td class='small'>residual of the matched crater centres (includes detection noise)</td></tr>")
    h.append(f"<tr><td>held-out crater RMSE</td><td>{m['heldout_rmse_ref_px']} (ref)</td><td>{m.get('heldout_rmse_src_orig_px')}</td><td class='small'>each crater predicted by a fit that never saw it</td></tr>")
    if m.get("dense_nodes"): h.append(f"<tr><td>image residual (RMS)</td><td>{m['dense_rms_ref_px']} (ref)</td><td>{m['dense_rms_src_orig_px']}</td><td class='small'>{m['dense_nodes']}/{m['dense_nodes_tried']} patches; median {m['dense_median_ref_px']}; within 0.5 px: {m['dense_within_0p5px']:.0%}; within 1 px: {m['dense_within_1px']:.0%}</td></tr>")
    if "rmse_metres" in m: h.append(f"<tr><td>in metres</td><td colspan='2'>crater RMSE {m['rmse_metres']} m" + (f"; image residual {m['dense_rms_metres']} m" if "dense_rms_metres" in m else "") + "</td><td class='small'>from the reference grid CSV</td></tr>")
    h.append("</table>")
    if P.get("accuracy") is not None: h.append(f"<h3>Residual shift map</h3><img src='data:image/jpeg;base64,{_img_b64(P['accuracy'])}'>")
    h.append("<p class='small'>Work pixels = pixels of the (possibly shrunk) copy the program works on; source-image pixels = the target image's original pixels. Grades: <= 0.5 px excellent, <= 1 px sub-pixel, <= 2 px near sub-pixel, <= 5 px pixel-level, above that coarse.</p>")
    h.append("<h2>2. Input metadata (XML labels)</h2>")
    for name, x in (("Reference", xr), ("Target / source", xt)):
        h.append(f"<h3>{name}: {html.escape(os.path.basename(x['file'] or '(none)'))}</h3>")
        h.append(f"<table>{_kv(x['facts'])}</table>" if x["facts"] else "<p class='small'>no XML given (or nothing readable)</p>")
    if R.get("xml_context"): h.append("<h3>Derived from both labels</h3><table>" + _kv(R["xml_context"]) + "</table><p class='small'>sun-angle fields appear only if the label contains them; the registration itself never uses them.</p>")
    h.append("<h2>3. Pattern evidence</h2>")
    for k, cap in (("lock", "Lock pattern: the same star (anchor crater + lines to its neighbours) in both images, and both stars normalised"), ("matches", "Matched craters (same colour = same crater)"), ("overlay", "Overlay (target in the red channel)"),
                   ("checker", "Checkerboard"), ("diff", "Difference of locally normalised images")):
        if P.get(k) is not None: h.append(f"<h3>{cap}</h3><img src='data:image/jpeg;base64,{_img_b64(P[k])}'>")
    if R.get("seeds"): h.append("<p>Saved patterns recognised in this pair: " + html.escape(", ".join(f"{a} (score {b})" for a, b in R["seeds"])) + "</p>")
    h.append("<h2>4. Comparison with plain SIFT + RANSAC</h2>")
    b = R.get("baseline")
    h.append(f"<table>{_kv(b)}</table><p class='small'>The baseline matches pixel appearance and is the usual solution; where it disagrees with the pattern result or finds few inliers, lighting / scale differences defeat appearance matching.</p>" if b else "<p class='small'>not run</p>")
    if R.get("stress"):
        h.append("<h2>5. Stress test with known ground truth</h2><table><tr><th>case</th><th>ok</th><th>craters</th><th>RMSE vs truth (px)</th><th>zoom found / true</th></tr>")
        for r in R["stress"]: h.append(f"<tr><td>{html.escape(r['case'])}</td><td>{r['ok']}</td><td>{r.get('craters_matched', '')}</td><td>{r.get('rmse_vs_truth_px', '')}</td><td>{r.get('zoom_found', '')} / {r.get('zoom_true', '')}</td></tr>")
        h.append("</table><p class='small'>The reference is registered against copies of itself with a known rotation / zoom and changed lighting (gamma, ramp, noise). Synthetic: a sanity check, not a substitute for real pairs.</p>")
    h.append("<h2>6. Problem-statement checklist</h2><table><tr><th>requirement</th><th>evidence in this run</th></tr>")
    for k, v in (("match points between source and reference", f"{m['n_inliers']} crater correspondences (matches.csv)"), ("registered product", "registered_target.png (target warped into the reference frame)"),
                 ("sub-pixel accuracy", f"crater centres ellipse-refined; RMSE {m['rmse_ref_px']} px, held-out {m['heldout_rmse_ref_px']} px (work pixels)"),
                 ("uniform distribution of matches", f"coverage {m['coverage_3x3']}, hull/overlap {m['hull_fraction_of_overlap']}"), ("sun-angle invariance", "pattern uses crater positions and sizes only"),
                 ("scale invariance", f"zoom x{m['zoom_tgt_over_ref']} recovered, no scale prior"), ("evaluation metrics", "RMSE, held-out RMSE, inlier count and ratio, coverage, uniqueness (metrics.json)")):
        h.append(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>")
    h.append("</table><h2>7. Settings and notes</h2><table>" + _kv({k: v for k, v in R["options"].items() if v not in (None, "")}) + "</table><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in R["notes"]) + "</ul>")
    for name, x in (("Reference", xr), ("Target", xt)):
        if x["table"]: h.append(f"<h2>Appendix: {name} XML (all fields)</h2><table>{''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k, v in x['table'])}</table>")
    h.append("</body></html>"); return "".join(h)


def build_pdf(R, path):
    try:
        import matplotlib; matplotlib.use("Agg"); from matplotlib.backends.backend_pdf import PdfPages; import matplotlib.pyplot as plt
    except Exception: return False
    m = R["metrics"]; P = R.get("products", {})
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.07, 0.95, "Crater-pattern registration report", fontsize=17, weight="bold", color="#0b3d91")
        lines = [f"reference: {os.path.basename(R['ref'])}", f"target:    {os.path.basename(R['tgt'])}", f"time: {R.get('seconds')} s", ""]
        if R["ok"]:
            for k in ("model", "n_inliers", "inlier_ratio", "zoom_tgt_over_ref", "rotation_deg", "rmse_ref_px", "heldout_rmse_ref_px", "coverage_3x3", "hull_fraction_of_overlap", "uniqueness", "rmse_metres"):
                if k in m: lines.append(f"{k:28s} {m[k]}")
            for nm, x in (("REFERENCE XML", R["xml_ref"]), ("TARGET XML", R["xml_tgt"])):
                lines += ["", nm] + [f"  {k}: {str(v)[:70]}" for k, v in x["facts"].items()]
            lines += [""] + [f"{k}: {v}" for k, v in R.get("xml_context", {}).items()]
            if R.get("baseline"): lines += ["", "SIFT+RANSAC baseline: " + json.dumps(R["baseline"])[:110]]
        else: lines.append(R.get("message", "not registered")[:100])
        fig.text(0.07, 0.92, "\n".join(lines), va="top", family="monospace", fontsize=8.5); pdf.savefig(fig); plt.close(fig)
        for k, cap in (("lock", "Lock pattern"), ("matches", "Matched craters"), ("overlay", "Overlay"), ("checker", "Checkerboard")):
            if P.get(k) is not None:
                fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.9]); ax.imshow(cv2.cvtColor(P[k], cv2.COLOR_BGR2RGB)); ax.axis("off"); fig.suptitle(cap); pdf.savefig(fig); plt.close(fig)
    return True


def save_results(R, out_dir=None):
    out = Path(out_dir or R["options"].get("out", OUT)); d = out / "registration" / f"{Path(R['ref']).stem}__{Path(R['tgt']).stem}"; (d / "steps").mkdir(parents=True, exist_ok=True); files = {}
    for i, (title, img, text) in enumerate(R["steps"], 1): cv2.imwrite(str(d / "steps" / f"{i:02d}.png"), img)
    (d / "steps.txt").write_text("\n\n".join(f"== {t}\n{x}" for t, _, x in R["steps"]))
    if R["ok"]:
        P = R["products"]; M = R["M"]; D = R["D"]; m = R["metrics"]
        for k, fn in (("registered", "registered_target.png"), ("overlay", "overlay.png"), ("checker", "checkerboard.png"), ("diff", "difference.png"), ("matches", "matches_overlay.png"), ("lock", "lock_pattern.png"), ("accuracy", "accuracy_map.png")):
            if P.get(k) is not None: cv2.imwrite(str(d / fn), P[k])
        T = np.array(R["T_ref_to_tgt"]); (d / "transform.json").write_text(json.dumps(dict(model=m["model"], T_reference_to_target=T.tolist(), T_target_to_reference=np.linalg.inv(T).tolist(), zoom_tgt_over_ref=m["zoom_tgt_over_ref"], rotation_deg=m["rotation_deg"],
                                                                                        note="3x3, WORK pixels (images shrunk to max_side); x_target = T @ x_reference", scale_ref=D["scA"], scale_tgt=D["scB"], original_size_ref=D["oA"], original_size_tgt=D["oB"]), indent=1))
        (d / "metrics.json").write_text(json.dumps({**m, "baseline": R.get("baseline"), "xml_context": R.get("xml_context"), "stress": R.get("stress"), "vs_official_csv_m": R.get("vs_official_csv_m"), "ecc": R.get("ecc")}, indent=1, default=str))
        pairs = M["pairs"]; A = M["PA"][[i for i, _ in pairs]]; B = M["PB"][[j for _, j in pairs]]; res = np.linalg.norm(apply_T(T, A[:, :2]) - B[:, :2], axis=1) / m["zoom_tgt_over_ref"]
        cols = dict(ref_x=A[:, 0], ref_y=A[:, 1], ref_r=radius(A), tgt_x=B[:, 0], tgt_y=B[:, 1], tgt_r=radius(B), residual_ref_px=res, ref_x_orig=(A[:, 0] + .5) / D["scA"] - .5, ref_y_orig=(A[:, 1] + .5) / D["scA"] - .5, tgt_x_orig=(B[:, 0] + .5) / D["scB"] - .5, tgt_y_orig=(B[:, 1] + .5) / D["scB"] - .5)
        if R.get("match_lonlat"): cols["lon"], cols["lat"] = R["match_lonlat"]
        import pandas as pd; pd.DataFrame(cols).round(4).to_csv(d / "matches.csv", index=False)
        if R.get("grid"):
            g = R["grid"]; ok = np.isfinite(g["Longitude"]) & np.isfinite(g["Latitude"]); stem = Path(R["tgt"]).stem.replace("_b_brw_", "_g_grd_"); stem = stem if "_g_grd_" in stem else stem + "_g_grd_est"
            pd.DataFrame(dict(Longitude=np.round(g["Longitude"][ok], 7), Latitude=np.round(g["Latitude"][ok], 7), Pixel=g["Pixel"][ok], Scan=g["Scan"][ok])).to_csv(d / f"{stem}.csv", index=False, lineterminator="\r\n"); files["grid_csv"] = str(d / f"{stem}.csv")
    if R["options"].get("make_report", True):
        (d / "report.html").write_text(build_report(R, d), encoding="utf-8"); files["report_html"] = str(d / "report.html")
        if build_pdf(R, str(d / "report.pdf")): files["report_pdf"] = str(d / "report.pdf")
    R["saved_dir"] = str(d); R["files"] = files; return d


# =============================================================== SEEK viewer: side by side  ->  glide onto each other by the pattern
def star_of(PA, pairs, k_show=7):
    ia = np.array([i for i, _ in pairs]); best, score = 0, -1
    for n in range(len(ia)):
        d = np.hypot(*(PA[ia, :2] - PA[ia[n], :2]).T); sc = (d < 4.0 * np.sort(d)[min(3, len(d) - 1)] + 1e-9).sum() + 0.02 * radius(PA[ia[n:n + 1]])[0]
        if sc > score: best, score = n, sc
    d = np.hypot(*(PA[ia, :2] - PA[ia[best], :2]).T); return best, [n for n in np.argsort(d) if n != best][:k_show]


class Seek:
    """u = 0: the two images next to each other.  u = 1: the target sits registered on the reference.  In between the target
    translates, rotates and zooms along the pattern (similarity of the final fit); the last 15 % cross-fade to the exact registered image."""
    def __init__(self, R):
        D, P = R["D"], R["products"]; self.R = R; self.A, self.B = D["imgA"], D["imgB"]; T = np.array(R["T_ref_to_tgt"]); self.Ti = np.linalg.inv(T)
        hA, wA = self.A.shape[:2]; hB, wB = self.B.shape[:2]; self.ct = np.array([wB / 2, hB / 2]); p0 = apply_T(self.Ti, self.ct[None])[0]; px = apply_T(self.Ti, (self.ct + [1, 0])[None])[0] - p0
        self.s1, self.th1, self.c1 = float(np.linalg.norm(px)), float(math.atan2(px[1], px[0])), p0; gap = 0.05 * max(wA, wB); self.c0 = np.array([wA + gap + wB / 2, hA / 2])
        self.b0 = np.array([0, min(0, (hA - hB) / 2), wA + gap + wB, max(hA, (hA + hB) / 2)], float)
        corners = apply_T(self.Ti, np.array([[0, 0], [wB, 0], [wB, hB], [0, hB]], float)); allp = np.vstack([corners, [[0, 0], [wA, 0], [wA, hA], [0, hA]]]); self.b1 = np.array([allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()], float)
        self.reg, self.valid = P["registered"], P["valid"].astype(np.uint8); M = R["M"]; self.PA, self.PB, self.pairs = M["PA"], M["PB"], M["pairs"]; self.a0, self.order = star_of(self.PA, self.pairs) if len(self.pairs) >= 3 else (0, [])
        self.cols = [tuple(int(x) for x in cv2.cvtColor(np.uint8([[[(k * 23) % 180, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for k in range(max(len(self.pairs), 1))]; self.banner = R["metrics"].get("quality_banner", "")
    def pose(self, u):
        s = math.exp(u * math.log(self.s1)); th = u * self.th1; c = (1 - u) * self.c0 + u * self.c1; Rm = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        return np.vstack([np.hstack([Rm, (c - Rm @ self.ct)[:, None]]), [0, 0, 1]]), s
    def view(self, u, cw, ch):
        b = (1 - u) * self.b0 + u * self.b1; sc = min(cw / (b[2] - b[0]), ch / (b[3] - b[1])) * 0.94
        return np.array([[sc, 0, (cw - sc * (b[2] - b[0])) / 2 - sc * b[0]], [0, sc, (ch - sc * (b[3] - b[1])) / 2 - sc * b[1]], [0, 0, 1]]), sc
    def render(self, t, cw=1000, ch=620, mode="blend", lines=True, circles=True, star=True, ease=True, op_ref=1.0, op_tgt=1.0, banner=True):
        u = t * t * (3 - 2 * t) if ease else t; Mv, sc = self.view(u, cw, ch); Pz, s = self.pose(u); Mt = Mv @ Pz; bg = (28, 28, 28)
        ones = lambda im: np.full(im.shape[:2], 255, np.uint8)
        refc = cv2.warpAffine(self.A, Mv[:2], (cw, ch), flags=cv2.INTER_LINEAR, borderValue=bg); mref = cv2.warpAffine(ones(self.A), Mv[:2], (cw, ch)) > 0
        tg = cv2.warpAffine(self.B, Mt[:2], (cw, ch), flags=cv2.INTER_LINEAR, borderValue=bg); at = (cv2.warpAffine(ones(self.B), Mt[:2], (cw, ch)) > 0).astype(np.float32)
        if u > 0.85:                                                                  # last 15 %: cross-fade to the exact registered image (affine / homography / refined)
            f = (u - 0.85) / 0.15; rg = cv2.warpAffine(self.reg, Mv[:2], (cw, ch), flags=cv2.INTER_LINEAR, borderValue=bg); mr = (cv2.warpAffine(self.valid * 255, Mv[:2], (cw, ch)) > 127).astype(np.float32)
            tg = np.where(mr[..., None] > 0, (1 - f) * tg + f * rg, tg).astype(np.uint8); at = (1 - f) * at + f * mr
        mrf = (mref.astype(np.float32) * op_ref)[..., None]; base = np.full_like(refc, 28) * (1 - mrf) + refc * mrf            # reference layer, own opacity
        a = at * op_tgt                                                                                                    # target layer, own opacity
        if mode == "checker":
            yy, xx = np.mgrid[0:ch, 0:cw]; chk = ((yy // 36 + xx // 36) % 2).astype(np.float32); a = a * np.where(mref, chk, 1.0)
        out = (base * (1 - a[..., None]) + tg * a[..., None]).astype(np.uint8)
        def cpt(M_, p): q = M_ @ np.array([p[0], p[1], 1.0]); return int(q[0]), int(q[1])
        if circles:
            for k, (i, j) in enumerate(self.pairs):
                cv2.circle(out, cpt(Mv, self.PA[i]), max(2, int(radius(self.PA[i:i + 1])[0] * sc)), self.cols[k], 2, cv2.LINE_AA); cv2.circle(out, cpt(Mt, self.PB[j]), max(2, int(radius(self.PB[j:j + 1])[0] * sc * s)), self.cols[k], 2, cv2.LINE_AA)
        if lines:
            for k, (i, j) in enumerate(self.pairs): cv2.line(out, cpt(Mv, self.PA[i]), cpt(Mt, self.PB[j]), self.cols[k], 1, cv2.LINE_AA)
        if star and self.order:
            ia, ib = [i for i, _ in self.pairs], [j for _, j in self.pairs]
            for P_, idx, M_ in ((self.PA, ia, Mv), (self.PB, ib, Mt)):
                for kk, n in enumerate(self.order): cv2.line(out, cpt(M_, P_[idx[self.a0]]), cpt(M_, P_[idx[n]]), (255, 255, 255), 3, cv2.LINE_AA); cv2.line(out, cpt(M_, P_[idx[self.a0]]), cpt(M_, P_[idx[n]]), self.cols[n % len(self.cols)], 1, cv2.LINE_AA)
                cv2.circle(out, cpt(M_, P_[idx[self.a0]]), 7, (0, 0, 255), -1, cv2.LINE_AA)
        txt = "separate images" if t < 0.02 else ("aligned by the crater pattern" if t > 0.98 else "moving along the pattern ...")
        tag(out, f"t = {t:.2f}   {txt}   (target scale x{s:.2f})", 22); tag(out, f"opacity: reference {op_ref * 100:.0f}%   target {op_tgt * 100:.0f}%", 44)
        if banner and self.banner: tag(out, self.banner, ch - 12)
        return out
    def save_gif(self, path, frames=60, size=(900, 560), **kw):
        from PIL import Image
        ims = [Image.fromarray(cv2.cvtColor(self.render(min(1, k / (frames - 1)), size[0], size[1], **kw), cv2.COLOR_BGR2RGB)) for k in range(frames)]; ims += [ims[-1]] * 12
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=45, loop=0); return path


# =============================================================== GUI
def run_gui(args):
    import tkinter as tk, threading, queue, webbrowser, glob
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    root = tk.Tk(); root.title("Crater Register v2  -  pattern based image registration"); root.geometry("1500x1000")
    keys = ("ref", "ref_xml", "ref_csv", "ref_craters", "tgt", "tgt_xml", "tgt_csv", "tgt_craters", "out"); V = {k: tk.StringVar() for k in keys}; V["out"].set(str(Path(args.out).resolve()))
    for k in ("ref", "tgt", "ref_xml", "tgt_xml", "ref_csv", "tgt_csv", "ref_craters", "tgt_craters"):
        if getattr(args, k, None): V[k].set(getattr(args, k))
    model = tk.StringVar(value="auto"); B = {k: tk.BooleanVar(value=DEFAULTS[k]) for k in ("use_ecc", "use_library", "use_model", "baseline", "stress", "make_report", "use_csv_craters")}
    SL = dict(min_r=tk.DoubleVar(value=DEFAULTS["min_r"]), ml_p=tk.DoubleVar(value=DEFAULTS["ml_p"]), k=tk.DoubleVar(value=DEFAULTS["k"]), max_craters=tk.DoubleVar(value=DEFAULTS["max_craters"]), max_side=tk.DoubleVar(value=DEFAULTS["max_side"]))
    S = dict(R=None, seek=None, photo=None, playing=False, u=0.0, steps_photo=None); q = queue.Queue()
    top = ttk.LabelFrame(root, text=" INPUT  (image required; XML / CSV optional) ", padding=4); top.pack(fill="x", padx=6, pady=3)
    IMG = [("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")]; XML = [("XML label", "*.xml"), ("All", "*.*")]; CSV = [("CSV", "*.csv"), ("All", "*.*")]
    def pick(key, types, side):
        p = filedialog.askdirectory() if types == "dir" else filedialog.askopenfilename(filetypes=types)
        if not p: return
        V[key].set(p)
        if key in ("ref", "tgt"):                                                         # auto-find the label / grid / crater files that sit next to the image
            stem = os.path.splitext(p)[0]; x = glob.glob(stem + ".xml"); g = glob.glob(stem.replace("_b_brw_", "_g_grd_") + ".csv"); c = glob.glob(str(Path(V["out"].get()) / Path(p).stem / "craters.csv"))
            if x and not V[key + "_xml"].get(): V[key + "_xml"].set(x[0])
            if g and not V[key + "_csv"].get(): V[key + "_csv"].set(g[0])
            if c and not V[key + "_craters"].get(): V[key + "_craters"].set(c[0])
    for r, (side, lab) in enumerate((("ref", "REFERENCE (fixed)"), ("tgt", "TARGET / SOURCE (moving)"))):
        ttk.Label(top, text=lab, width=24).grid(row=r, column=0, sticky="w")
        for c, (suffix, hint, types) in enumerate((("", "image", IMG), ("_xml", "XML", XML), ("_csv", "geo CSV", CSV), ("_craters", "craters CSV", CSV))):
            f = ttk.Frame(top); f.grid(row=r, column=1 + c, padx=2, pady=1); ttk.Entry(f, textvariable=V[side + suffix], width=21).pack(side="left"); ttk.Button(f, text=hint + "...", width=len(hint) + 1, command=lambda k=side + suffix, t=types: pick(k, t, None)).pack(side="left")
    f = ttk.Frame(top); f.grid(row=2, column=1, columnspan=2, sticky="w", pady=2); ttk.Label(f, text="Output folder").pack(side="left"); ttk.Entry(f, textvariable=V["out"], width=60).pack(side="left", padx=4); ttk.Button(f, text="...", width=3, command=lambda: pick("out", "dir", None)).pack(side="left")
    opt = ttk.LabelFrame(root, text=" OPTIONS (advanced) ", padding=4); opt.pack(fill="x", padx=6, pady=2); opt1 = ttk.Frame(opt); opt1.pack(fill="x")
    ttk.Label(opt1, text="Transform").pack(side="left"); ttk.Combobox(opt1, textvariable=model, width=11, state="readonly", values=["auto", "similarity", "affine", "homography"]).pack(side="left", padx=3)
    for k, t in (("use_ecc", "dense sub-pixel (ECC)"), ("use_library", "use saved patterns"), ("use_model", "use trained model"), ("use_csv_craters", "use crater CSVs"), ("baseline", "SIFT baseline"), ("stress", "stress test"), ("make_report", "report")): ttk.Checkbutton(opt1, text=t, variable=B[k]).pack(side="left", padx=3)
    opt2 = ttk.Frame(opt); opt2.pack(fill="x")
    for k, lab, lo, hi, res in (("min_r", "min radius", 3, 40, 1), ("ml_p", "ML conf", 0.1, 0.9, 0.05), ("k", "neighbours k", 4, 14, 1), ("max_craters", "max craters", 40, 300, 10), ("max_side", "work size px", 600, 2400, 100)):
        ttk.Label(opt2, text=lab).pack(side="left", padx=(6, 0)); tk.Scale(opt2, from_=lo, to=hi, resolution=res, orient="horizontal", length=110, variable=SL[k], showvalue=True).pack(side="left")
    runb = tk.Button(opt1, text="  RUN MATCHING  ", bg="#2a7", fg="white", font=("Arial", 12, "bold")); runb.pack(side="right", padx=6)
    nb = ttk.Notebook(root); nb.pack(fill="both", expand=True, padx=6, pady=3)
    # ---- seek tab
    seek_tab = ttk.Frame(nb); nb.add(seek_tab, text="Side by side -> aligned (seek bar)"); qlab = tk.Label(seek_tab, text="  run the matching to see RMSE and the sub-pixel verdict here", font=("Arial", 12, "bold"), anchor="w", bg="#222", fg="#ccc"); qlab.pack(fill="x"); cv = tk.Canvas(seek_tab, bg="#1c1c1c", highlightthickness=0); cv.pack(fill="both", expand=True)
    ctl = ttk.Frame(seek_tab); ctl.pack(fill="x"); play = ttk.Button(ctl, text="Play", width=11); play.pack(side="left", padx=3); useek = tk.DoubleVar(value=0.0); speed = tk.DoubleVar(value=1.0); loop = tk.BooleanVar(value=False)
    bar = tk.Scale(ctl, from_=0, to=1, resolution=0.002, orient="horizontal", variable=useek, length=700, showvalue=False); bar.pack(side="left", fill="x", expand=True, padx=6)
    ttk.Label(ctl, text="speed").pack(side="left"); tk.Scale(ctl, from_=0.2, to=4, resolution=0.2, orient="horizontal", variable=speed, length=70, showvalue=False).pack(side="left")
    flags = dict(lines=tk.BooleanVar(value=True), circles=tk.BooleanVar(value=True), star=tk.BooleanVar(value=True)); mode = tk.StringVar(value="blend")
    for k, t in (("lines", "pair lines"), ("circles", "craters"), ("star", "lock pattern")): ttk.Checkbutton(ctl, text=t, variable=flags[k], command=lambda: show()).pack(side="left", padx=2)
    ttk.Checkbutton(ctl, text="loop", variable=loop).pack(side="left"); ttk.Combobox(ctl, textvariable=mode, width=8, state="readonly", values=["blend", "checker"]).pack(side="left", padx=3)
    ttk.Button(ctl, text="Reset", width=6, command=lambda: set_u(0)).pack(side="left"); gifb = ttk.Button(ctl, text="Save GIF", width=9); gifb.pack(side="left", padx=3)
    ctl2 = ttk.Frame(seek_tab); ctl2.pack(fill="x"); opr = tk.DoubleVar(value=100); opt_ = tk.DoubleVar(value=100); flick = tk.BooleanVar(value=False)
    ttk.Label(ctl2, text="Reference opacity %").pack(side="left", padx=(6, 0)); tk.Scale(ctl2, from_=0, to=100, orient="horizontal", variable=opr, length=170, command=lambda v: show()).pack(side="left")
    ttk.Label(ctl2, text="Target opacity %").pack(side="left", padx=(10, 0)); tk.Scale(ctl2, from_=0, to=100, orient="horizontal", variable=opt_, length=170, command=lambda v: show()).pack(side="left")
    for t_, a_, b_ in (("50 / 50", 50, 50), ("100 / 100", 100, 100), ("reference only", 100, 0), ("target only", 0, 100)): ttk.Button(ctl2, text=t_, command=lambda a_=a_, b_=b_: (opr.set(a_), opt_.set(b_), show())).pack(side="left", padx=2)
    ttk.Checkbutton(ctl2, text="flicker target (overlap check)", variable=flick).pack(side="left", padx=8)
    # ---- steps tab
    st_tab = ttk.Frame(nb); nb.add(st_tab, text="Steps"); lb = tk.Listbox(st_tab, width=52, font=("Arial", 11)); lb.pack(side="left", fill="y"); right = ttk.Frame(st_tab); right.pack(side="left", fill="both", expand=True)
    sc_ = tk.Canvas(right, bg="#1c1c1c", highlightthickness=0); sc_.pack(fill="both", expand=True); stxt = tk.Text(right, height=6, wrap="word", font=("Arial", 11)); stxt.pack(fill="x")
    met = tk.Text(nb, font=("Menlo", 11)); nb.add(met, text="Metrics / report"); logt = tk.Text(nb, font=("Menlo", 10), bg="#1b1b1b", fg="#9fe"); nb.add(logt, text="Log")
    def say(s): q.put(("log", s))
    def opts():
        o = {k: v.get() for k, v in B.items()}; o.update(model=model.get(), min_r=SL["min_r"].get(), ml_p=SL["ml_p"].get(), k=int(SL["k"].get()), max_craters=int(SL["max_craters"].get()), max_side=int(SL["max_side"].get()), out=V["out"].get())
        for k in ("ref_xml", "tgt_xml", "ref_csv", "tgt_csv", "ref_craters", "tgt_craters"): o[k] = V[k].get() or None
        return {**DEFAULTS, **o}
    def work():
        try:
            global OUT; OUT = Path(V["out"].get()); R = register_pair(V["ref"].get(), V["tgt"].get(), opts(), log=say)
            if R["ok"] or R["steps"]: save_results(R, V["out"].get()); say(f"saved -> {R.get('saved_dir')}")
            q.put(("done", R))
        except Exception:
            import traceback; q.put(("err", traceback.format_exc()))
    def run():
        if not V["ref"].get() or not V["tgt"].get(): messagebox.showwarning("Missing", "Select the reference and the target image."); return
        logt.delete("1.0", "end"); runb.config(state="disabled"); nb.select(logt); S["playing"] = False; threading.Thread(target=work, daemon=True).start()
    runb.config(command=run)
    def show(_=None):
        if S["seek"] is None: return
        cw, ch = max(cv.winfo_width(), 200), max(cv.winfo_height(), 150); ot = 0.0 if (flick.get() and S.get("flick_off")) else opt_.get() / 100
        fr = S["seek"].render(float(useek.get()), cw, ch, mode.get(), flags["lines"].get(), flags["circles"].get(), flags["star"].get(), op_ref=opr.get() / 100, op_tgt=ot)
        S["photo"] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))); cv.delete("all"); cv.create_image(0, 0, anchor="nw", image=S["photo"])
        play.config(text="Pause" if S["playing"] else ("Play again" if useek.get() >= 0.999 else "Play"))
    def set_u(u): useek.set(u); show()
    bar.config(command=lambda v: show()); cv.bind("<Configure>", lambda e: show())
    def restart():
        if S["seek"] is not None and loop.get(): useek.set(0.0); S["playing"] = True; show()
    def tick():
        if S["playing"] and S["seek"] is not None:
            u = useek.get() + 0.006 * speed.get()
            if u >= 1.0:
                u = 1.0; S["playing"] = False                                  # STOPS at the end; the button turns into 'Play again'
                if loop.get(): root.after(900, restart)
            useek.set(u); show()
        elif flick.get() and S["seek"] is not None:
            S["fl"] = S.get("fl", 0) + 1
            if S["fl"] % 10 == 0: S["flick_off"] = not S.get("flick_off", False); show()
        root.after(40, tick)
    def toggle():
        if S["seek"] is None: return
        if S["playing"]: S["playing"] = False
        else:
            if useek.get() >= 0.999: useek.set(0.0)                            # 'Play again' starts from the beginning
            S["playing"] = True
        show()
    play.config(command=toggle)
    def gif():
        if S["seek"] is None: return
        d = Path(S["R"].get("saved_dir", V["out"].get())); d.mkdir(parents=True, exist_ok=True); p = S["seek"].save_gif(str(d / "seek_animation.gif"), mode=mode.get(), lines=flags["lines"].get(), circles=flags["circles"].get(), star=flags["star"].get()); say(f"GIF saved: {p}"); messagebox.showinfo("GIF", p)
    gifb.config(command=gif)
    def step_show(_=None):
        sel = lb.curselection()
        if not sel or not S["R"]: return
        t, img, txt = S["R"]["steps"][sel[0]]; cw, ch = max(sc_.winfo_width(), 100), max(sc_.winfo_height(), 100); k = min(cw / img.shape[1], ch / img.shape[0]); im = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        S["steps_photo"] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))); sc_.delete("all"); sc_.create_image(0, 0, anchor="nw", image=S["steps_photo"]); stxt.delete("1.0", "end"); stxt.insert("end", t + "\n\n" + txt)
    lb.bind("<<ListboxSelect>>", step_show)
    def fill_metrics(R):
        met.delete("1.0", "end"); m = R["metrics"]; lines = ["RESULT: " + ("REGISTERED" if R["ok"] else "NOT REGISTERED"), ""]
        if R["ok"]: lines += [f"{k:28s} {v}" for k, v in m.items() if k != "subpixel_note"] + ["", "Baseline SIFT+RANSAC: " + json.dumps(R.get("baseline")), "XML context: " + json.dumps(R.get("xml_context")), "vs official CSV (m): " + json.dumps(R.get("vs_official_csv_m")), "Stress test: " + json.dumps(R.get("stress")), "", "Saved patterns recognised: " + json.dumps(R.get("seeds"))]
        else: lines.append(R.get("message", ""))
        lines += ["", "Notes:"] + ["  " + n for n in R["notes"]] + ["", "Files: " + json.dumps(R.get("files"), indent=1), "", f"Report: {R.get('files', {}).get('report_html', '(not made)')}"]; met.insert("end", "\n".join(lines))
    def openrep():
        p = (S["R"] or {}).get("files", {}).get("report_html")
        if p: webbrowser.open("file://" + os.path.abspath(p))
    ttk.Button(root, text="Open report in browser", command=openrep).pack(anchor="e", padx=8)
    def poll():
        try:
            while True:
                k, d = q.get_nowait()
                if k == "log": logt.insert("end", d + "\n"); logt.see("end")
                elif k == "err": logt.insert("end", d); runb.config(state="normal"); messagebox.showerror("Failed", d.strip().splitlines()[-1])
                else:
                    runb.config(state="normal"); S["R"] = d; lb.delete(0, "end")
                    for t, _, _ in d["steps"]: lb.insert("end", t)
                    fill_metrics(d)
                    if d["ok"]: S["seek"] = Seek(d); qlab.config(text="  " + d["metrics"]["quality_banner"], fg=d["metrics"]["quality_color"]); set_u(0.0); nb.select(seek_tab); root.after(300, show)
                    else: nb.select(met); messagebox.showinfo("No match", d.get("message", ""))
        except queue.Empty: pass
        root.after(120, poll)
    poll(); tick()
    globals()["_HOOK"] = dict(play=play, opr=opr, opt_=opt_, flick=flick, toggle=toggle, qlab=qlab, useek=useek, root=root, S=S, V=V, run=run, set_u=set_u, nb=nb, seek_tab=seek_tab, say=say)
    if args.ref and args.tgt: root.after(400, run)
    root.mainloop()


# =============================================================== command line
def main():
    global OUT
    ap = argparse.ArgumentParser(description="Crater Register v2")
    ap.add_argument("--ref"); ap.add_argument("--tgt", nargs="*", help="one or more target images (batch)"); ap.add_argument("--out", default="output")
    for k in ("ref_xml", "tgt_xml", "ref_csv", "tgt_csv", "ref_craters", "tgt_craters"): ap.add_argument("--" + k.replace("_", "-"), dest=k)
    ap.add_argument("--model", default="auto", choices=["auto", "similarity", "affine", "homography"]); ap.add_argument("--no-ecc", action="store_true"); ap.add_argument("--no-library", action="store_true"); ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--stress", action="store_true", help="also run the known-ground-truth stress test"); ap.add_argument("--gif", action="store_true"); ap.add_argument("--gui", action="store_true"); ap.add_argument("--max-side", type=int, default=DEFAULTS["max_side"])
    ap.add_argument("--min-r", type=float, default=DEFAULTS["min_r"]); ap.add_argument("--conf", type=float, default=DEFAULTS["ml_p"]); a = ap.parse_args(); OUT = Path(a.out); CLN.Store(OUT)
    tg = a.tgt or []
    if a.gui or not (a.ref and tg):
        a.tgt = tg[0] if tg else None; return run_gui(a)
    rows = []
    for t in tg:
        O = {**DEFAULTS, **dict(out=a.out, model=a.model, use_ecc=not a.no_ecc, use_library=not a.no_library, baseline=not a.no_baseline, stress=a.stress, max_side=a.max_side, min_r=a.min_r, ml_p=a.conf)}
        for k in ("ref_xml", "tgt_xml", "ref_csv", "tgt_csv", "ref_craters", "tgt_craters"): O[k] = getattr(a, k)
        R = register_pair(a.ref, t, O); d = save_results(R, a.out); print("saved ->", d)
        if a.gif and R["ok"]: print("gif ->", Seek(R).save_gif(str(d / "seek_animation.gif")))
        rows.append(dict(target=Path(t).name, ok=R["ok"], **{k: R["metrics"].get(k) for k in ("model", "n_inliers", "inlier_ratio", "zoom_tgt_over_ref", "rotation_deg", "rmse_ref_px", "heldout_rmse_ref_px", "coverage_3x3")}))
    if len(rows) > 1:
        import pandas as pd; pd.DataFrame(rows).to_csv(Path(a.out) / "batch_summary.csv", index=False); print("batch summary ->", Path(a.out) / "batch_summary.csv")


if __name__ == "__main__":
    main()