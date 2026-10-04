CRATER SUITE  -  put these 3 files in one folder
  crater_lens.py       your original detector (unchanged)
  crater_learn.py      TRAIN + detect craters (GUI)           python crater_learn.py
  crater_register.py   PATTERN matching + overlay (GUI)       python crater_register.py --ref big_area.png --tgt zoomed.png

pip install numpy opencv-python pillow scipy scikit-learn joblib

1) Train (crater_learn.py)
   Open image -> "Auto-label" (free starting craters) -> fix: tool 'crater' = drag centre->rim, 'NOT crater' = red, 'erase'
   -> Train ML -> Detect (ML) -> tools 'accept'/'reject' on detections -> Train ML again.   (Self-train is optional.)
   Open several images (different light / zoom) and train on all of them: labels are kept in output/annotations.csv.
   Saved: output/model.joblib, output/annotations.csv, output/<image>/craters.csv + craters.json + overlay.png
   !! If you have an OLD output/model.joblib from an earlier version, delete it and train again (features changed).

2) Match (crater_register.py)
   Uses output/model.joblib if it exists; if not, it trains itself on the strong craters of the two images.
   Steps 1-10 are shown in the GUI; 6b is the lock pattern (same star in both images).
   Saved (Save results -> output/registration/<ref>__<tgt>/): transform.json, warped_target.png, overlay.png, checkerboard.png, steps.
   Headless: python crater_register.py --ref A.png --tgt B.png --export
