# 🌙 LunaAlign: Crater-Pattern Registration Engine

**"Matching the Geometry of the Moon, Not Its Appearance"**

[![Powered By](https://img.shields.io/badge/Powered%20By-OpenCV%20%7C%20scikit--learn-blue?style=for-the-badge&logo=python)](https://opencv.org/)
[![Problem Statement](https://img.shields.io/badge/ISRO-Chandrayaan--2%20Image%20Registration-orange?style=for-the-badge)](https://www.isro.gov.in/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-lightgrey?style=for-the-badge)](#-limitations)

---
## The Vision
A Chandrayaan-2 image and an LRO reference of the same ground rarely look alike. The **Sun angle**, **viewing geometry**, **sensor** and **resolution** all change, so appearance-based matchers (SIFT, ORB, learned matchers) return few, clustered or false matches, and a high match count says little about accuracy.

**LunaAlign** matches **craters** instead. Crater centres and sizes stay put when the shadows move, and the *pattern* a crater forms with its neighbours does not change with zoom, rotation or shift. We turn that into a registration engine that outputs a registered image, match points and honest quality metrics.

## Why Crater Patterns? (The LunaAlign Edge)
Appearance matching asks *"do these pixels look alike?"*. LunaAlign asks *"is this the same arrangement of craters?"*.

* **Sun-angle proof:** uses crater **positions and sizes**, never brightness. The detector reads rotation-invariant ring features, so the light direction does not matter.
* **Zoom and rotation proof:** the pattern descriptor uses **ratios**, so scale is recovered, not assumed.
* **Trainable:** pen-draw the craters it missed and mark what is *not* a crater. It learns your terrain.
* **Honest accuracy:** reports crater RMSE, held-out RMSE, and the residual measured on the **image pixels**, in the source image's own units.

| Problem-statement challenge | LunaAlign answer |
| :--- | :--- |
| Illumination variation | Matches crater geometry; detection on locally normalised, rotation-invariant features |
| Scale variation | Lock-pattern triangle **ratios**; zoom found automatically |
| Viewpoint variation | Similarity / affine / homography, chosen by leave-one-out error |
| Sub-pixel accuracy | Ellipse-refined centres + image correlation with parabolic peak fit |
| Uniform distribution | Coverage and hull metrics, regular patch grid |
| Evaluation metrics | RMSE, held-out RMSE, inlier count and ratio, coverage, residual map, SIFT baseline, stress test |

## Architecture
1.  **Ingestion:** reference and target images, with optional XML labels, geo grid CSVs and crater CSVs.
2.  **Crater Detector:** an ExtraTrees classifier on polar ring features, with a dense multi-scale scan and sub-pixel ellipse refinement.
3.  **Lock Patterns:** every crater plus two of its *k* nearest neighbours forms a triangle with a zoom / rotation / shift-invariant descriptor. Patterns you saved also seed the search.
4.  **Verification:** a hypothesis wins by lining up the most **whole craters** (position **and** size), counting only craters that should be visible in both frames.
5.  **Model and Refinement:** the transform is chosen by leave-one-out error, with optional dense ECC refinement.
6.  **Accuracy Check and Outputs:** registered image, `matches.csv`, `metrics.json`, an HTML/PDF report and a lon/lat grid CSV.

```mermaid
flowchart LR
    A[Reference + Target<br/>optional XML · geo CSV · crater CSV] --> B[Illumination-tolerant<br/>preprocessing]
    B --> C[Trainable crater detector]
    C --> D[Sub-pixel ellipse refinement]
    D --> E[Lock-pattern triangles]
    L[(Saved patterns)] -.seeds.-> F
    E --> F[Hypothesis voting<br/>zoom · rotation · shift]
    F --> G[Whole-crater verification]
    G --> H[Model choice<br/>leave-one-out]
    H --> I[Dense refinement ECC]
    I --> J[Accuracy check<br/>patch correlation]
    J --> K[Registered image · matches.csv<br/>metrics · report · lon/lat grid]
```

## File Structure
```text
LunaAlign/
├── crater_lens.py          # Classic detector (Hough + sub-pixel edge fit)
├── crater_learn.py         # Teach Layer: train craters, build / save / locate patterns (GUI)
├── crater_register.py      # Match Layer: pattern registration, viewer, report (GUI + CLI)
├── README.md
└── output/                 # Generated
    ├── model.joblib        # Trained crater model
    ├── annotations.csv     # Your labels
    ├── patterns.json       # Saved crater patterns
    └── registration/<ref>__<target>/
        ├── registered_target.png · overlay.png · checkerboard.png · accuracy_map.png
        ├── matches.csv · metrics.json · transform.json
        ├── <target>_g_grd_est.csv    # ISRO-format grid for the target
        └── report.html · report.pdf
```

## Quick Start
```bash
pip install numpy opencv-python pillow scipy scikit-learn joblib pandas   # matplotlib optional (PDF report)

python crater_learn.py        # 1. teach it your craters, save 1-4 patterns per image
python crater_register.py     # 2. pick reference + target, press RUN MATCHING
```

Headless:
```bash
python crater_register.py --ref ref.png --tgt a.png b.png --out output \
       --ref-xml ref.xml --tgt-xml a.xml --ref-csv ref_g_grd.csv --stress --gif
```

## The Live Viewer
`crater_register.py` shows both images **side by side** with a **seek bar**. Slide or press **Play**: the target glides, rotates and zooms onto the reference along the crater pattern, and the lock-pattern star ends up on top of itself.

* Per-image **opacity**, 50/50, reference-only, target-only and **flicker** overlap checks
* **Zoom** (wheel, buttons, pixel view above 600%), drag to pan, double-click to fit
* Coloured **accuracy bar** with crater RMSE, held-out RMSE, image residual and a verdict
* 10 explained steps and **GIF export**

## Metrics
| Metric | Meaning |
| :--- | :--- |
| `n_inliers`, `inlier_ratio` | Craters matched / craters that should be visible in both |
| `rmse_ref_px` | Residual of matched crater centres |
| `heldout_rmse_*` | Leave-one-out error: each crater predicted by a fit that never saw it |
| `dense_rms_src_orig_px` | Residual shift measured on the images, in source-image pixels |
| `coverage_3x3`, `hull_fraction_of_overlap` | Spatial distribution of the matches |
| `quality` | ≤0.5 px excellent · ≤1 px sub-pixel · ≤2 near sub-pixel · ≤5 pixel-level · else coarse |

## Results So Far
>  **All numbers are from synthetic lunar-like terrain with known ground truth. They are not results on real Chandrayaan-2 / LRO pairs.**

| Test (target lit from the opposite side) | Result |
| :--- | :--- |
| 1.6× zoom, 70° rotation | 20 craters matched · zoom 1.6001 (true 1.6) · ≈0.45 px vs truth |
| Same pair, plain SIFT + RANSAC | 6 inliers, disagrees with the truth by hundreds of px |
| Rotate 25°, zoom 0.85 | 44 craters · 0.19 px |
| Rotate 80°, zoom 0.55 (zoomed out) | 26 craters · 0.27 px |
| Rotate −50°, zoom 1.6 (zoomed in) | 17 craters · 3.3 px |
| Patterns found, Sun elevation 10°–70° | 4 / 4, anchor error < 1 px |

## Limitations
* Needs **≥ 4 craters** matched in both images; a strong zoom-in showing only 2–3 craters fails.
* **Very thin strips** (the 120 px IIRS and 168 px TMC browse images) are not handled yet; IIRS also needs destriping.
* Not yet validated on real OHRC / TMC / IIRS ↔ LRO pairs. GeoTIFF references are untested.
* Similarity / affine / homography only; no terrain-height model for strong parallax.

## Roadmap
- [ ] Strip mode: native-resolution windows, IIRS destriping, TMC shadow handling
- [ ] Validate on real Chandrayaan-2 ↔ LRO NAC / SELENE pairs
- [ ] Extra candidate generators (RIFT2, SuperPoint + LightGlue, LoFTR) feeding the same verification stage
- [ ] Phase-correlation sub-pixel refinement and reliability-weighted point selection
