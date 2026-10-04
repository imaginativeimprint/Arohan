<div align="center">

# 🌙 LunaAlign

**Crater-pattern image registration for Chandrayaan-2 (OHRC · TMC-2 · IIRS) and lunar reference data**

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![status](https://img.shields.io/badge/status-research%20prototype-orange)
![validation](https://img.shields.io/badge/validated%20on-synthetic%20terrain-yellow)

*Match the **geometry** of the Moon, not its **appearance** — so Sun angle, zoom and rotation stop mattering.*

</div>

---

## Why this exists

Registering a Chandrayaan-2 image to an LRO NAC / SELENE reference is hard because the same terrain looks different when the **Sun angle**, **viewing geometry**, **sensor** and **resolution** change. Appearance-based matchers (SIFT, ORB, learned matchers) then return few, clustered or false correspondences, and a high match count says little about accuracy.

LunaAlign matches **craters** instead. Crater centres and sizes stay put when shadows move, and the *pattern* a crater forms with its neighbours is unchanged by zoom, rotation and shift.

| Challenge (problem statement) | How LunaAlign addresses it |
|---|---|
| Illumination / Sun-angle variation | Matches crater **positions and sizes**, never brightness. Detection uses local-contrast-normalised, rotation-invariant ring features |
| Scale variation | Lock-pattern triangles use **ratios** → zoom is recovered, not assumed |
| Viewpoint variation | Similarity / affine / homography chosen by leave-one-out error |
| Sub-pixel accuracy | Ellipse-refined crater centres + image correlation with parabolic peak fit |
| Uniform distribution of match points | Coverage and hull metrics; regular patch grid for the accuracy check |
| Evaluation metrics | RMSE, held-out RMSE, inlier count / ratio, coverage, residual map, SIFT baseline, stress test |

---

## Pipeline

```mermaid
flowchart LR
    A[Reference + Target images<br/>optional XML · geo CSV · crater CSV] --> B[Illumination-tolerant<br/>preprocessing]
    B --> C[Trainable crater detector<br/>ExtraTrees on ring features]
    C --> D[Sub-pixel ellipse refinement]
    D --> E[Lock-pattern triangles<br/>anchor + k nearest neighbours]
    L[(Saved patterns<br/>patterns.json)] -.seeds.-> F
    E --> F[Hypothesis voting<br/>zoom · rotation · shift]
    F --> G[Verification<br/>whole craters line up<br/>position AND size]
    G --> H[Model choice by<br/>leave-one-out error]
    H --> I[Optional dense refinement<br/>ECC]
    I --> J[Accuracy check<br/>patch correlation + parabola]
    J --> K[Registered image · matches.csv<br/>metrics · report · lon/lat grid]
    style C fill:#e8f1ff,stroke:#2f6fdb
    style E fill:#e8f1ff,stroke:#2f6fdb
    style J fill:#e9f9ee,stroke:#2e9e57
```

```mermaid
flowchart TD
    subgraph Teach[" crater_learn.py : teach it your craters "]
        t1[Open image] --> t2[Auto-label from classic detector]
        t2 --> t3[Draw / erase / mark NOT-crater]
        t3 --> t4[Train ML] --> t5[Detect] --> t6{Happy?}
        t6 -- no: accept / reject --> t3
        t6 -- yes --> t7[Propose 1-4 patterns<br/>OK · Edit · Draw · Delete]
        t7 --> t8[(model.joblib<br/>annotations.csv<br/>patterns.json)]
    end
    t8 --> reg[crater_register.py : register two images]
```

---

## Key ideas

**1. Trainable crater detector (`crater_learn.py`)**
Each candidate circle is sampled in polar form (24 angles × 10 radii) on a scale-adaptive image pyramid, then described by ring statistics and **FFT magnitudes over angle**, which makes the description rotation-invariant (so Sun azimuth does not matter) and size-independent (so zoom does not matter). An ExtraTrees classifier is trained from your pen-drawn craters, craters marked *not-crater*, and auto-labels. A dense multi-scale scan finds rims that Hough voting misses.

**2. Lock patterns (`crater_register.py`)**
Every crater plus two of its *k* nearest neighbours forms a triangle. Its descriptor is unchanged by zoom, rotation and shift:

```python
# side ratios + crater-size ratios  ->  5-D invariant descriptor
desc = [s_short / s_long,  s_mid / s_long,  r_v0 / s_long,  r_v1 / s_long,  r_v2 / s_long]
```

Look-alike triangles in the two images each propose a similarity transform.

**3. Verification by whole craters**
A proposal is scored by how many craters land on a crater of the **same position and size**. Only craters that *should* be visible in the other frame are counted, so a small zoomed-in view is not punished for craters outside it.

**4. Honest accuracy**
Crater-centre RMSE includes detection noise, so LunaAlign also correlates patches of the registered target against the reference, refines each peak with a parabola, rejects edge-of-window false matches, and reports the **residual shift measured on the images** in the *source image's own pixels*.

---

## Quick start

```bash
pip install numpy opencv-python pillow scipy scikit-learn joblib pandas   # matplotlib optional (PDF report)
```

Keep these three files in one folder (`crater_lens.py` is the original Crater Lens detector; `crater_learn.py` finds it automatically):

```
.
├── crater_lens.py       # classic detector (Hough + sub-pixel edge fit)
├── crater_learn.py      # train craters · build / save / locate patterns   (GUI)
└── crater_register.py   # register two images by crater pattern             (GUI + CLI)
```

```bash
python crater_learn.py                         # 1. teach it your craters, save patterns
python crater_register.py                      # 2. pick reference + target, RUN MATCHING
```

Headless:

```bash
python crater_register.py --ref ref.png --tgt a.png b.png --out output \
       --ref-xml ref.xml --tgt-xml a.xml --ref-csv ref_g_grd.csv --stress --gif
```

<details>
<summary><b>All inputs and options</b></summary>

| Input | Required | Used for |
|---|---|---|
| Reference image, target image | ✔ | registration |
| `.xml` PDS4 label (per image) | ✖ | metadata, acquisition-time and Sun-angle differences in the report |
| geo grid CSV (`Longitude,Latitude,Pixel,Scan`) | ✖ | lon/lat of matches, RMSE in metres, generated grid CSV for the target |
| crater CSV (from `crater_learn`) | ✖ | skip detection |

Options: transform (`auto / similarity / affine / homography`), dense sub-pixel ECC, saved patterns, trained model, SIFT baseline, stress test, report, min radius, ML confidence, neighbours *k*, max craters, work size.

</details>

---

## The viewer

`crater_register.py` opens with both images **side by side** and a **seek bar**. Slide or press **Play**: the target glides, rotates and zooms onto the reference along the crater pattern, lines join matching craters, and the lock-pattern star ends up on top of itself. Play stops at the end (**Play again**).

- per-image **opacity** sliders, 50/50, reference-only, target-only and **flicker** for overlap checks
- **zoom** (wheel / buttons / pixel view above 600%), drag to pan, double-click to fit
- coloured **accuracy bar**: crater RMSE, held-out RMSE, image residual and a verdict
- 10 explained **steps**, **GIF export**

---

## Outputs

```
output/registration/<ref>__<target>/
├── registered_target.png   overlay.png   checkerboard.png   difference.png
├── matches_overlay.png     lock_pattern.png   accuracy_map.png
├── matches.csv             # crater correspondences (+ lon/lat if a geo CSV was given)
├── metrics.json            transform.json
├── <target>_g_grd_est.csv  # ISRO-format grid for the target (needs the reference geo CSV)
└── report.html  report.pdf # metadata, metrics, evidence, baseline, stress test, PS checklist
```

| Metric | Meaning |
|---|---|
| `n_inliers`, `inlier_ratio` | craters matched / craters that should be visible in both |
| `rmse_ref_px` | residual of matched crater centres |
| `heldout_rmse_*` | leave-one-out: each crater predicted by a fit that never saw it |
| `dense_rms_src_orig_px` | residual shift measured on the images, in source-image pixels |
| `coverage_3x3`, `hull_fraction_of_overlap` | spatial distribution of the matches |
| `uniqueness` | best zoom hypothesis vs best competing zoom |
| `quality` | ≤0.5 px excellent · ≤1 px sub-pixel · ≤2 near sub-pixel · ≤5 pixel-level · else coarse |

---

## Results so far

> ⚠️ **All numbers below are from synthetic lunar-like terrain with known ground truth. They are not results on real Chandrayaan-2 / LRO pairs.**

| Test (target lit from the opposite side) | Result |
|---|---|
| 1.6× zoom, 70° rotation | 20 craters matched · zoom 1.6001 (true 1.6) · ≈0.45 px error vs truth |
| Same pair, plain SIFT + RANSAC | 6 inliers, disagrees with the truth by hundreds of px |
| Stress: rotate 25°, zoom 0.85 | 44 craters · 0.19 px |
| Stress: rotate 80°, zoom 0.55 (zoomed out) | 26 craters · 0.27 px |
| Stress: rotate −50°, zoom 1.6 (zoomed in) | 17 craters · 3.3 px |
| Patterns found across Sun elevations 10°–70° | 4 / 4, anchor error < 1 px |
| Detector, 10 drawn craters | finds 33 / 35 undrawn craters (classic pipeline: 25) |
| Generated grid CSV vs official (synthetic grid) | 0.85 m mean |

---

## Limitations

- Needs **≥ 4 craters** matched in both images; a strong zoom-in that shows only 2–3 craters fails.
- **Very thin strips** (e.g. the 120 px wide IIRS and 168 px wide TMC browse images) are not handled yet: the default work size shrinks them to a few tens of pixels. IIRS also needs destriping.
- Similarity / affine / homography only; no terrain-height model for strong parallax.
- Not yet validated on real OHRC / TMC / IIRS ↔ LRO pairs. GeoTIFF references are untested.
- Errors are reported in work pixels and converted to source-image pixels; check both.
- CPU implementation.

## Roadmap

- [ ] **Strip mode**: native-resolution windows along the strip, IIRS destriping, TMC shadow handling
- [ ] Validate on real Chandrayaan-2 ↔ LRO NAC / SELENE pairs
- [ ] Extra candidate generators (RIFT2, SuperPoint + LightGlue, LoFTR) feeding the same verification stage
- [ ] Phase-correlation / upsampled-DFT sub-pixel refinement
- [ ] Reliability-weighted selection of control points
- [ ] Terrain-aware model (DEM relighting)

---

## Repository layout

```
crater_register.py   register_pair() · knn_triangles() · hypothesis_match() · dense_check() · Seek viewer · report
crater_learn.py      FeatPyr · scan_candidates() · train_model() · detect_ml() · find_pattern() · propose_patterns()
crater_lens.py       classic Hough + sub-pixel ellipse detector (input to the trainer)
```

## Acknowledgements

Built for the ISRO problem statement *"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)"*. Data: Chandrayaan-2 (ISSDC), LRO NAC (LROC), SELENE.

## License

Add your licence here (e.g. MIT).
