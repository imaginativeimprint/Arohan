#!/usr/bin/env python3
"""
Planetary Image Analyzer & Crater Detector (GPU Optimized)
============================================================
Native desktop app (Tkinter + OpenCV + NumPy/Pandas, optional PyTorch
MPS/CUDA acceleration). No web components.

Run:
    pip install opencv-python pillow numpy pandas torch scikit-image
    python planetary_analyzer.py
"""

import os
import time
import threading
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageTk

try:
    from skimage import measure as _skimage_measure
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

# --------------------------------------------------------------------------
# Optional GPU backend
# --------------------------------------------------------------------------
try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def resolve_device():
    if not _TORCH_OK:
        return None, "CPU (torch not installed)"
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps"), "Apple GPU (MPS)"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return torch.device("cuda"), "NVIDIA GPU (CUDA)"
    except Exception:
        pass
    return torch.device("cpu"), "CPU (torch fallback)"


DEVICE, DEVICE_LABEL = resolve_device()

DEFAULT_CONTRAST = 2.2
DEFAULT_THRESHOLD = 15
DEFAULT_BLUR = 10


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass
class Params:
    method: str = "Pencil"
    contrast: float = DEFAULT_CONTRAST
    threshold: int = DEFAULT_THRESHOLD
    blur: int = DEFAULT_BLUR
    theme: str = "Normal"
    grid_on: bool = False
    grid_opacity: float = 0.6
    grid_color_bgr: tuple = (0, 255, 0)


@dataclass
class Telemetry:
    lons: np.ndarray = None
    lats: np.ndarray = None
    lon_grid: np.ndarray = None
    lat_grid: np.ndarray = None
    pix_unique: np.ndarray = None
    scan_unique: np.ndarray = None
    is_regular_grid: bool = False


@dataclass
class Crater:
    cx: int
    cy: int
    r: int
    circularity: float


# --------------------------------------------------------------------------
# Core image-processing engine
# --------------------------------------------------------------------------
class PlanetaryEngine:

    def __init__(self, device):
        self.device = device

    def _empty_cache(self):
        if self.device is None:
            return
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    def gaussian_blur(self, gray_f32, radius):
        ksize = max(1, int(radius) * 2 + 1)

        if self.device is not None:
            try:
                t = torch.from_numpy(gray_f32).to(self.device)[None, None, :, :]
                sigma = max(ksize / 3.0, 0.5)
                half = ksize // 2
                xs = torch.arange(-half, half + 1, dtype=torch.float32, device=self.device)
                g1d = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
                g1d = g1d / g1d.sum()
                t = torch.nn.functional.conv2d(t, g1d.view(1, 1, 1, -1), padding=(0, half))
                t = torch.nn.functional.conv2d(t, g1d.view(1, 1, -1, 1), padding=(half, 0))
                out = t[0, 0].detach().to("cpu").numpy()
                del t
                self._empty_cache()
                return out
            except Exception:
                pass
        return cv2.GaussianBlur(gray_f32, (ksize, ksize), 0)

    def dodge_blend(self, base_f32, blend_f32):
        if self.device is not None:
            try:
                b = torch.from_numpy(base_f32).to(self.device)
                bl = torch.from_numpy(blend_f32).to(self.device)
                result = (b * 255.0) / (255.0 - bl + 1.0)
                result = torch.clamp(result, 0, 255)
                out = result.detach().to("cpu").numpy()
                del b, bl, result
                self._empty_cache()
                return out
            except Exception:
                pass
        result = (base_f32 * 255.0) / (255.0 - blend_f32 + 1.0)
        return np.clip(result, 0, 255)

    def apply_contrast(self, img_f32, contrast):
        if self.device is not None:
            try:
                t = torch.from_numpy(img_f32).to(self.device)
                out = torch.clamp(t * float(contrast), 0, 255).detach().to("cpu").numpy()
                del t
                self._empty_cache()
                return out
            except Exception:
                pass
        return np.clip(img_f32 * float(contrast), 0, 255)

    def apply_method(self, gray_f32, method, blur_value, contrast, threshold):
        g8 = np.clip(gray_f32, 0, 255).astype(np.uint8)

        if method == "Sobel":
            sx = cv2.Sobel(g8, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(g8, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(sx, sy)
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            inverted = 255.0 - mag
            return self.apply_contrast(inverted.astype(np.float32), contrast)

        if method == "Laplacian":
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            lap = cv2.filter2D(g8.astype(np.float32), -1, kernel)
            lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
            inverted = 255.0 - lap
            sharpened = cv2.filter2D(
                inverted, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            )
            sharpened = np.clip(sharpened, 0, 255)
            return self.apply_contrast(sharpened.astype(np.float32), contrast)

        if method == "Canny":
            blurred = cv2.GaussianBlur(g8, (3, 3), 0)
            lo = max(5, int(threshold))
            hi = min(255, lo * 3)
            edges = cv2.Canny(blurred, lo, hi)
            inverted = 255.0 - edges.astype(np.float32)
            return self.apply_contrast(inverted, contrast)

        if method == "Pencil":
            inv = 255.0 - gray_f32
            blurred = self.gaussian_blur(inv, blur_value)
            sketch = self.dodge_blend(gray_f32, blurred)
            sketch = self.apply_contrast(sketch, contrast)
            sketch = self._professional_shading(sketch)
            return sketch

        return gray_f32

    def _professional_shading(self, gray_f32):
        try:
            g8 = np.clip(gray_f32, 0, 255).astype(np.uint8)
            bilateral = cv2.bilateralFilter(g8, d=9, sigmaColor=75, sigmaSpace=75)
            kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=np.float32)
            embossed = cv2.filter2D(bilateral, -1, kernel)
            blended = cv2.addWeighted(bilateral, 0.7, embossed, 0.3, 0)
            if int(blended.max()) != int(blended.min()):
                blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX)
            return blended.astype(np.float32)
        except Exception:
            return gray_f32

    def apply_threshold_floor(self, img_f32, threshold):
        if threshold <= 0:
            return img_f32
        return np.where(img_f32 < threshold, 0, img_f32).astype(np.float32)

    def apply_theme(self, gray_f32, theme):
        g8 = np.clip(gray_f32, 0, 255).astype(np.uint8)

        if theme == "Normal":
            return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

        if theme == "Negative":
            return cv2.cvtColor(255 - g8, cv2.COLOR_GRAY2BGR)

        if theme == "Depth (3D Emboss)":
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
            embossed = cv2.filter2D(g8.astype(np.float32), -1, kernel) + 128
            embossed = np.clip(embossed, 0, 255).astype(np.uint8)
            return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

        if theme == "Temperature (Cold)":
            color = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR).astype(np.float32)
            color[:, :, 0] *= 1.25
            color[:, :, 2] *= 0.85
            return np.clip(color, 0, 255).astype(np.uint8)

        if theme == "Temperature (Warm)":
            color = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR).astype(np.float32)
            color[:, :, 2] *= 1.25
            color[:, :, 0] *= 0.85
            return np.clip(color, 0, 255).astype(np.uint8)

        if theme == "Shadow Detect":
            _, shadow_mask = cv2.threshold(g8, 60, 255, cv2.THRESH_BINARY_INV)
            color = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
            color[shadow_mask > 0] = (255, 120, 0)
            return color

        return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

    def detect_craters(self, gray_f32):
        g8 = np.clip(gray_f32, 0, 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(g8, (15, 15), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        h, w = g8.shape
        max_area = h * w * 0.35
        kernel = np.ones((5, 5), np.uint8)
        craters = []

        for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            _, thresh = cv2.threshold(enhanced, 0, 255, flag + cv2.THRESH_OTSU)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if not (150 < area < max_area):
                    continue
                perimeter = cv2.arcLength(c, True)
                if perimeter <= 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if not (0.35 < circularity <= 1.0):
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(c)
                craters.append(Crater(cx=int(cx), cy=int(cy), r=int(r), circularity=circularity))

        deduped = []
        for cr in sorted(craters, key=lambda c: -c.circularity):
            if any(abs(cr.cx - d.cx) < d.r and abs(cr.cy - d.cy) < d.r for d in deduped):
                continue
            deduped.append(cr)

        return deduped

    def compute_sun_angle(self, source_gray_f32, craters):
        if not craters:
            return float("nan"), float("nan")

        src8 = np.clip(source_gray_f32, 0, 255).astype(np.uint8)
        h, w = src8.shape
        angles = []

        for c in craters:
            cx, cy, r = c.cx, c.cy, max(c.r, 1)
            x0, y0 = max(cx - r, 0), max(cy - r, 0)
            x1, y1 = min(cx + r, w - 1), min(cy + r, h - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
            masked_for_search = np.where(mask > 0, src8, 255)
            _minVal, _maxVal, minLoc, _maxLoc = cv2.minMaxLoc(masked_for_search)
            dx = minLoc[0] - cx
            dy = minLoc[1] - cy
            if dx == 0 and dy == 0:
                continue
            angle = (np.degrees(np.arctan2(dx, -dy)) + 360.0) % 360.0
            angles.append(angle)

        if not angles:
            return float("nan"), float("nan")

        sin_sum = np.sum(np.sin(np.radians(angles)))
        cos_sum = np.sum(np.cos(np.radians(angles)))
        sun_angle = (np.degrees(np.arctan2(sin_sum, cos_sum)) + 360.0) % 360.0
        shadow_angle = (sun_angle + 180.0) % 360.0
        return sun_angle, shadow_angle

    def render_grid_overlay(self, base_bgr, telemetry, opacity, color_bgr, n_lines=6):
        h, w = base_bgr.shape[:2]
        if telemetry.lons is None or telemetry.lats is None:
            return base_bgr
        if len(telemetry.lons) == 0 or len(telemetry.lats) == 0:
            return base_bgr

        overlay = np.zeros_like(base_bgr)
        labels = []

        if telemetry.is_regular_grid and _SKIMAGE_OK:
            self._draw_curved_grid(overlay, labels, telemetry, w, h, n_lines)
        else:
            self._draw_straight_grid(overlay, labels, telemetry, w, h, n_lines)

        if self.device is not None:
            try:
                base_t = torch.from_numpy(base_bgr.astype(np.float32)).to(self.device)
                over_t = torch.from_numpy(overlay.astype(np.float32)).to(self.device)
                blended = base_t * (1.0 - opacity) + over_t * opacity
                blended = torch.where(over_t.sum(dim=-1, keepdim=True) > 0, blended, base_t)
                out = blended.detach().to("cpu").numpy()
                del base_t, over_t, blended
                self._empty_cache()
                result = np.clip(out, 0, 255).astype(np.uint8)
            except Exception:
                result = cv2.addWeighted(base_bgr, 1.0, overlay, opacity, 0)
        else:
            result = cv2.addWeighted(base_bgr, 1.0, overlay, opacity, 0)

        for (x, y, text) in labels:
            cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color_bgr, 1, cv2.LINE_AA)

        return result

    @staticmethod
    def _nice_ticks(vmin, vmax, n):
        if vmax <= vmin:
            return np.array([vmin])
        return np.round(np.linspace(vmin, vmax, n), 3)

    def _draw_straight_grid(self, overlay, labels, telemetry, w, h, n_lines):
        lon_min, lon_max = float(np.min(telemetry.lons)), float(np.max(telemetry.lons))
        lat_min, lat_max = float(np.min(telemetry.lats)), float(np.max(telemetry.lats))

        if lon_max > lon_min:
            for lon in self._nice_ticks(lon_min, lon_max, n_lines):
                x = int((lon - lon_min) / (lon_max - lon_min) * (w - 1))
                cv2.line(overlay, (x, 0), (x, h - 1), (200, 200, 200), 1, lineType=cv2.LINE_AA)
                labels.append((min(x + 4, w - 60), 16, f"{lon:.2f}E"))

        if lat_max > lat_min:
            for lat in self._nice_ticks(lat_min, lat_max, n_lines):
                y = int((lat - lat_min) / (lat_max - lat_min) * (h - 1))
                cv2.line(overlay, (0, y), (w - 1, y), (200, 200, 200), 1, lineType=cv2.LINE_AA)
                labels.append((4, max(min(y - 4, h - 4), 12), f"{lat:.2f}N"))

    def _draw_curved_grid(self, overlay, labels, telemetry, w, h, n_lines):
        lon_grid = telemetry.lon_grid
        lat_grid = telemetry.lat_grid
        pix_unique = telemetry.pix_unique
        scan_unique = telemetry.scan_unique
        n_scan_pts, n_pix_pts = lon_grid.shape

        pixel_max = float(pix_unique.max())
        scan_max = float(scan_unique.max())
        sx = (w - 1) / pixel_max if pixel_max > 0 else 1.0
        sy = (h - 1) / scan_max if scan_max > 0 else 1.0

        row_idx = np.arange(n_scan_pts)
        col_idx = np.arange(n_pix_pts)

        def draw_family(grid, ticks, suffix):
            for tick in ticks:
                try:
                    contours = _skimage_measure.find_contours(grid - tick, level=0)
                except Exception:
                    continue
                for path in contours:
                    if len(path) < 2:
                        continue
                    rows, cols = path[:, 0], path[:, 1]
                    scan_vals = np.interp(rows, row_idx, scan_unique)
                    pix_vals = np.interp(cols, col_idx, pix_unique)
                    xs = (pix_vals * sx).astype(np.int32)
                    ys = (scan_vals * sy).astype(np.int32)
                    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
                    cv2.polylines(overlay, [pts], isClosed=False, color=(200, 200, 200),
                                  thickness=1, lineType=cv2.LINE_AA)
                    mid = len(pts) // 2
                    mx, my = int(pts[mid, 0, 0]), int(pts[mid, 0, 1])
                    mx = max(2, min(mx, w - 60))
                    my = max(12, min(my, h - 4))
                    labels.append((mx, my, f"{tick:.2f}{suffix}"))

        lon_min, lon_max = float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid))
        lat_min, lat_max = float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))
        lon_ticks = self._nice_ticks(lon_min, lon_max, n_lines + 2)[1:-1]
        lat_ticks = self._nice_ticks(lat_min, lat_max, n_lines + 2)[1:-1]

        draw_family(lon_grid, lon_ticks, "E")
        draw_family(lat_grid, lat_ticks, "N")


# --------------------------------------------------------------------------
# Tkinter application
# --------------------------------------------------------------------------
class PlanetaryAnalyzerApp:
    METHODS = ["Sobel", "Laplacian", "Canny", "Pencil"]
    THEMES = ["Normal", "Negative", "Depth (3D Emboss)", "Temperature (Cold)",
              "Temperature (Warm)", "Shadow Detect"]

    def __init__(self, root):
        self.root = root
        self.root.title("Planetary Image Analyzer & Crater Detector")
        self.root.geometry("1300x800")
        self.root.minsize(1100, 700)
        self.root.configure(bg='#1a1a1a')

        self.engine = PlanetaryEngine(DEVICE)
        self.params = Params()
        self.telemetry = Telemetry()

        self.original_bgr = None
        self.gray_f32 = None
        self.processed_gray_f32 = None
        self.sketch_image = None
        self.display_photo = None

        self.detected_craters = []
        self.craters_visible = False
        self.sun_angle = float("nan")
        self.shadow_angle = float("nan")

        self._canvas_size = (860, 640)
        self._job_lock = threading.Lock()
        self._job_id = 0
        self._debounce_after_id = None

        self._build_ui()
        self._log(f"Ready. Compute device: {DEVICE_LABEL}")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Configure styles
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        
        # Custom styles for dark theme
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"), 
                       foreground="#00ccff", background="#1a1a1a")
        style.configure("Subtitle.TLabel", font=("Helvetica", 10), 
                       foreground="#888888", background="#1a1a1a")
        style.configure("Dark.TFrame", background="#1a1a1a")
        style.configure("Dark.TLabel", background="#1a1a1a", foreground="#e0e0e0")
        style.configure("Dark.TLabelframe", background="#1a1a1a", foreground="#e0e0e0")
        style.configure("Dark.TLabelframe.Label", background="#1a1a1a", 
                       foreground="#00ccff", font=("Helvetica", 10, "bold"))
        style.configure("Dark.TButton", background="#2d2d2d", foreground="#e0e0e0",
                       borderwidth=1, focuscolor="none")
        style.map("Dark.TButton",
                 background=[("active", "#3d3d3d"), ("pressed", "#4d4d4d")])
        style.configure("Dark.TCheckbutton", background="#1a1a1a", foreground="#e0e0e0")
        style.configure("Dark.TCombobox", fieldbackground="#2d2d2d", 
                       foreground="#e0e0e0", background="#2d2d2d")
        style.configure("Dark.TScale", background="#1a1a1a", troughcolor="#2d2d2d")
        # ttk quirk: Scale/Progressbar look up their layout as
        # "Horizontal.<StyleName>" / "Vertical.<StyleName>" specifically —
        # a custom style name does NOT automatically fall back to the
        # built-in TScale layout the way other widgets do. Without this,
        # the very first ttk.Scale(..., style="Dark.TScale") raises:
        #   _tkinter.TclError: Layout Horizontal.Dark.TScale not found
        style.layout("Horizontal.Dark.TScale", style.layout("Horizontal.TScale"))

        # Main container
        main = ttk.Frame(self.root, padding=10, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        # Left Panel
        panel = ttk.Frame(main, width=350, style="Dark.TFrame")
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        panel.pack_propagate(False)

        # Title Section
        title_frame = ttk.Frame(panel, style="Dark.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="🚀 Planetary Analyzer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_frame, text="Advanced Crater Detection & Analysis", 
                 style="Subtitle.TLabel").pack(anchor="w")

        # Scrollable panel for controls
        canvas_frame = ttk.Frame(panel, style="Dark.TFrame")
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas with scrollbar for controls
        control_canvas = tk.Canvas(canvas_frame, bg="#1a1a1a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=control_canvas.yview)
        scrollable_frame = ttk.Frame(control_canvas, style="Dark.TFrame")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        control_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Input Section
        io_frame = ttk.LabelFrame(scrollable_frame, text="📁 Input", padding=10, 
                                 style="Dark.TLabelframe")
        io_frame.pack(fill=tk.X, pady=4)
        
        upload_btn = ttk.Button(io_frame, text="📤 Upload Image", command=self.upload_image,
                               style="Dark.TButton")
        upload_btn.pack(fill=tk.X, pady=2)
        
        csv_btn = ttk.Button(io_frame, text="📊 Load Telemetry CSV", command=self.load_csv,
                            style="Dark.TButton")
        csv_btn.pack(fill=tk.X, pady=2)
        
        self.csv_status_var = tk.StringVar(value="No telemetry loaded")
        ttk.Label(io_frame, textvariable=self.csv_status_var, foreground="#888888",
                 style="Dark.TLabel").pack(anchor="w", pady=(4, 0))

        # Filters Section
        filt_frame = ttk.LabelFrame(scrollable_frame, text="🎨 Filters", padding=10,
                                   style="Dark.TLabelframe")
        filt_frame.pack(fill=tk.X, pady=4)

        # Method selection
        ttk.Label(filt_frame, text="Method:", style="Dark.TLabel").pack(anchor="w")
        self.method_var = tk.StringVar(value=self.params.method)
        method_box = ttk.Combobox(filt_frame, textvariable=self.method_var, 
                                 values=self.METHODS, state="readonly", 
                                 style="Dark.TCombobox")
        method_box.pack(fill=tk.X, pady=(0, 6))
        method_box.bind("<<ComboboxSelected>>", lambda e: self._on_change())

        # Sliders
        self.contrast_var = tk.DoubleVar(value=self.params.contrast)
        self._make_slider(filt_frame, "Contrast", self.contrast_var, 0.1, 3.0, "{:.1f}")

        self.threshold_var = tk.IntVar(value=self.params.threshold)
        self._make_slider(filt_frame, "Threshold", self.threshold_var, 0, 200, "{}")

        self.blur_var = tk.IntVar(value=self.params.blur)
        self._make_slider(filt_frame, "Blur", self.blur_var, 1, 20, "{}")

        # Theme Section
        theme_frame = ttk.LabelFrame(scrollable_frame, text="🌈 Theme", padding=10,
                                    style="Dark.TLabelframe")
        theme_frame.pack(fill=tk.X, pady=4)
        
        self.theme_var = tk.StringVar(value=self.params.theme)
        theme_box = ttk.Combobox(theme_frame, textvariable=self.theme_var, 
                                values=self.THEMES, state="readonly",
                                style="Dark.TCombobox")
        theme_box.pack(fill=tk.X)
        theme_box.bind("<<ComboboxSelected>>", lambda e: self._on_change())

        # Grid Section
        grid_frame = ttk.LabelFrame(scrollable_frame, text="🗺️ Geospatial Grid", padding=10,
                                   style="Dark.TLabelframe")
        grid_frame.pack(fill=tk.X, pady=4)
        
        self.grid_on_var = tk.BooleanVar(value=self.params.grid_on)
        grid_check = ttk.Checkbutton(grid_frame, text="Show Grid", variable=self.grid_on_var,
                                    command=self._on_change, style="Dark.TCheckbutton")
        grid_check.pack(anchor="w")
        
        self.grid_opacity_var = tk.DoubleVar(value=self.params.grid_opacity)
        self._make_slider(grid_frame, "Opacity", self.grid_opacity_var, 0.1, 1.0, "{:.1f}")

        color_row = ttk.Frame(grid_frame, style="Dark.TFrame")
        color_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(color_row, text="🎨 Grid Color", command=self._pick_grid_color,
                  style="Dark.TButton").pack(side=tk.LEFT)
        self.color_swatch = tk.Canvas(color_row, width=24, height=18, 
                                     highlightthickness=1, highlightbackground="#555",
                                     bg="#00ff00")
        self.color_swatch.pack(side=tk.LEFT, padx=6)

        # Detection Section
        detect_frame = ttk.LabelFrame(scrollable_frame, text="🔍 Detection", padding=10,
                                     style="Dark.TLabelframe")
        detect_frame.pack(fill=tk.X, pady=4)
        
        detect_btn = ttk.Button(detect_frame, text="🔴 Detect Craters", 
                               command=self.detect_craters, style="Dark.TButton")
        detect_btn.pack(fill=tk.X, pady=2)
        
        sun_btn = ttk.Button(detect_frame, text="☀️ Detect Sun Angle", 
                            command=self.detect_sun_angle, style="Dark.TButton")
        sun_btn.pack(fill=tk.X, pady=2)

        # Action Buttons
        action_frame = ttk.Frame(scrollable_frame, style="Dark.TFrame")
        action_frame.pack(fill=tk.X, pady=4)
        
        reset_btn = ttk.Button(action_frame, text="🔄 Reset All", command=self.reset_all,
                              style="Dark.TButton")
        reset_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        
        download_btn = ttk.Button(action_frame, text="💾 Download", 
                                 command=self.download_image, style="Dark.TButton")
        download_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # Status Log
        status_frame = ttk.LabelFrame(scrollable_frame, text="📋 Status Log", padding=10,
                                     style="Dark.TLabelframe")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        
        log_container = ttk.Frame(status_frame, style="Dark.TFrame")
        log_container.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_container, height=8, wrap="word", state="disabled",
                               bg="#0d0d0d", fg="#00ff88", font=("Consolas", 9),
                               relief=tk.FLAT, borderwidth=0)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical", 
                                     command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        # Right Panel - Image Display
        right = ttk.Frame(main, style="Dark.TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas with border
        canvas_container = ttk.Frame(right, style="Dark.TFrame")
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg="#0a0a0a", highlightthickness=1,
                               highlightbackground="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Status Bar
        bar = ttk.Frame(right, style="Dark.TFrame")
        bar.pack(fill=tk.X, pady=(6, 0))
        
        self.stats_var = tk.StringVar(value=self._format_status())
        ttk.Label(bar, textvariable=self.stats_var, font=("Consolas", 10),
                 foreground="#888888", style="Dark.TLabel").pack(side=tk.LEFT)
        
        device_label = ttk.Label(bar, text=f"⚡ {DEVICE_LABEL}", 
                                font=("Consolas", 9), foreground="#00ccff",
                                style="Dark.TLabel")
        device_label.pack(side=tk.RIGHT)

    def _make_slider(self, parent, label, var, frm, to, format_str):
        """Create a slider with live value display and continuous updates."""
        row = ttk.Frame(parent, style="Dark.TFrame")
        row.pack(fill=tk.X, pady=(4, 0))
        
        # Label row
        label_frame = ttk.Frame(row, style="Dark.TFrame")
        label_frame.pack(fill=tk.X)
        
        ttk.Label(label_frame, text=label, style="Dark.TLabel").pack(side=tk.LEFT)
        
        # Value display
        val_display = tk.StringVar()
        if isinstance(var, tk.DoubleVar):
            val_display.set(format_str.format(var.get()))
        else:
            val_display.set(format_str.format(var.get()))
        
        value_label = ttk.Label(label_frame, textvariable=val_display, 
                               font=("Consolas", 9), foreground="#00ccff",
                               style="Dark.TLabel")
        value_label.pack(side=tk.RIGHT)
        
        def on_slider_change(value):
            # Update display
            if isinstance(var, tk.DoubleVar):
                val_display.set(format_str.format(float(value)))
            else:
                val_display.set(format_str.format(int(float(value))))
            
            # Trigger debounced update
            if self._debounce_after_id is not None:
                self.root.after_cancel(self._debounce_after_id)
            self._debounce_after_id = self.root.after(10, self._on_change)
        
        # Create slider with continuous updates
        slider = ttk.Scale(row, from_=frm, to=to, variable=var, orient=tk.HORIZONTAL,
                          command=on_slider_change, style="Dark.TScale")
        slider.pack(fill=tk.X, pady=(2, 0))
        
        # Update on variable change (for reset)
        def on_var_change(*args):
            if isinstance(var, tk.DoubleVar):
                val_display.set(format_str.format(var.get()))
            else:
                val_display.set(format_str.format(var.get()))
        
        var.trace_add("write", on_var_change)

    def _pick_grid_color(self):
        rgb, _ = colorchooser.askcolor(title="Choose grid color")
        if rgb:
            r, g, b = [int(c) for c in rgb]
            self.params.grid_color_bgr = (b, g, r)
            self.color_swatch.config(bg=f"#{r:02x}{g:02x}{b:02x}")
            self._on_change()

    # ------------------------------------------------------------- loading
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select planetary image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), 
                      ("All files", "*.*")],
        )
        if not path:
            return
        try:
            pil_img = Image.open(path).convert("RGB")
            arr = np.array(pil_img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # Resize if too large
            max_dim = 1600
            h, w = bgr.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), 
                               interpolation=cv2.INTER_AREA)

            self.original_bgr = bgr
            self.gray_f32 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            self.detected_craters = []
            self.craters_visible = False
            self.sun_angle = float("nan")
            self.shadow_angle = float("nan")

            # Reset controls
            self.contrast_var.set(DEFAULT_CONTRAST)
            self.threshold_var.set(DEFAULT_THRESHOLD)
            self.blur_var.set(DEFAULT_BLUR)
            self.method_var.set("Pencil")
            self.theme_var.set("Normal")
            self.grid_on_var.set(False)

            self._log(f"✅ Image loaded: {os.path.basename(path)} ({bgr.shape[1]}x{bgr.shape[0]})")
            self._on_change()
        except Exception as exc:
            messagebox.showerror("Image load failed", str(exc))
            self._log(f"❌ ERROR loading image: {exc}")

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select telemetry CSV",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            cols = {c.strip().lower(): c for c in df.columns}
            required = ["longitude", "latitude", "pixel", "scan"]
            missing = [r for r in required if r not in cols]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            df[cols["longitude"]] = pd.to_numeric(df[cols["longitude"]], errors="coerce")
            df[cols["latitude"]] = pd.to_numeric(df[cols["latitude"]], errors="coerce")
            df[cols["pixel"]] = pd.to_numeric(df[cols["pixel"]], errors="coerce")
            df[cols["scan"]] = pd.to_numeric(df[cols["scan"]], errors="coerce")
            df = df.dropna(subset=[cols["longitude"]], how="any")

            lons = df[cols["longitude"]].to_numpy()
            lats = df[cols["latitude"]].to_numpy()

            telemetry = Telemetry(lons=lons, lats=lats)

            # Try to reshape into grid
            try:
                pix_unique = np.sort(df[cols["pixel"]].unique())
                scan_unique = np.sort(df[cols["scan"]].unique())
                expected_rows = len(pix_unique) * len(scan_unique)
                if 0 < len(df) and abs(expected_rows - len(df)) / expected_rows < 0.01:
                    piv_lon = df.pivot_table(index=cols["scan"], columns=cols["pixel"],
                                              values=cols["longitude"])
                    piv_lat = df.pivot_table(index=cols["scan"], columns=cols["pixel"],
                                              values=cols["latitude"])
                    piv_lon = piv_lon.reindex(index=scan_unique, columns=pix_unique)
                    piv_lat = piv_lat.reindex(index=scan_unique, columns=pix_unique)
                    lon_grid = piv_lon.to_numpy()
                    lat_grid = piv_lat.to_numpy()
                    nan_frac = np.isnan(lon_grid).mean()
                    if nan_frac < 0.02:
                        telemetry.lon_grid = lon_grid
                        telemetry.lat_grid = lat_grid
                        telemetry.pix_unique = pix_unique
                        telemetry.scan_unique = scan_unique
                        telemetry.is_regular_grid = True
            except Exception:
                pass

            self.telemetry = telemetry
            self.csv_status_var.set(f"✅ {len(lons):,} telemetry rows loaded")

            if self.telemetry.is_regular_grid:
                self._log(f"✅ Telemetry CSV loaded: {os.path.basename(path)} "
                         f"({len(lons):,} rows, full geolocation grid)")
                if not _SKIMAGE_OK:
                    self._log("⚠️ Install scikit-image for curved grid lines")
            else:
                self._log(f"✅ Telemetry CSV loaded: {os.path.basename(path)} "
                         f"({len(lons):,} rows, straight-line approximation)")
            self._on_change()
        except Exception as exc:
            messagebox.showerror("CSV load failed", str(exc))
            self._log(f"❌ ERROR loading CSV: {exc}")

    # --------------------------------------------------------- change loop
    def _on_slider_change(self):
        if self._debounce_after_id is not None:
            self.root.after_cancel(self._debounce_after_id)
        self._debounce_after_id = self.root.after(10, self._on_change)

    def _on_change(self):
        if self.gray_f32 is None:
            return

        self.params.method = self.method_var.get()
        self.params.contrast = float(self.contrast_var.get())
        self.params.threshold = int(self.threshold_var.get())
        self.params.blur = int(self.blur_var.get())
        self.params.theme = self.theme_var.get()
        self.params.grid_on = bool(self.grid_on_var.get())
        self.params.grid_opacity = float(self.grid_opacity_var.get())

        if self._debounce_after_id is not None:
            self.root.after_cancel(self._debounce_after_id)
            self._debounce_after_id = None

        with self._job_lock:
            self._job_id += 1
            job_id = self._job_id

        threading.Thread(target=self._run_pipeline, args=(job_id,), daemon=True).start()

    def _on_canvas_resize(self, event):
        self._canvas_size = (max(event.width, 100), max(event.height, 100))
        if self.original_bgr is not None:
            self._on_change()

    # ------------------------------------------------------------ pipeline
    def _run_pipeline(self, job_id):
        try:
            t0 = time.perf_counter()
            gray = self.gray_f32
            p = self.params

            filtered = self.engine.apply_method(gray, p.method, p.blur, p.contrast, p.threshold)
            processed = self.engine.apply_threshold_floor(filtered, p.threshold)

            themed_bgr = self.engine.apply_theme(processed, p.theme)

            if self.craters_visible and self.detected_craters:
                for c in self.detected_craters:
                    cv2.circle(themed_bgr, (c.cx, c.cy), c.r, (0, 100, 255), 2, 
                              lineType=cv2.LINE_AA)
                    cv2.circle(themed_bgr, (c.cx, c.cy), 2, (0, 255, 255), -1,
                              lineType=cv2.LINE_AA)

            if p.grid_on:
                themed_bgr = self.engine.render_grid_overlay(
                    themed_bgr, self.telemetry, p.grid_opacity, p.grid_color_bgr
                )

            full_res_rgb = cv2.cvtColor(themed_bgr, cv2.COLOR_BGR2RGB)
            full_res_pil = Image.fromarray(full_res_rgb)

            cw, ch = self._canvas_size
            h, w = themed_bgr.shape[:2]
            scale = min(cw / w, ch / h) if w and h else 1.0
            scale = max(scale, 0.01)
            disp = cv2.resize(themed_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_LANCZOS4)
            disp_pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            self.root.after(0, self._apply_result, job_id, processed, 
                          full_res_pil, disp_pil, elapsed_ms)
        except Exception:
            err = traceback.format_exc()
            self.root.after(0, self._log, f"❌ ERROR in pipeline:\n{err}")

    def _apply_result(self, job_id, processed_gray_f32, full_res_pil, disp_pil, elapsed_ms):
        with self._job_lock:
            if job_id != self._job_id:
                return

        self.processed_gray_f32 = processed_gray_f32
        self.sketch_image = full_res_pil

        self.display_photo = ImageTk.PhotoImage(disp_pil)
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or self._canvas_size[0]
        ch = self.canvas.winfo_height() or self._canvas_size[1]
        self.canvas.create_image(cw // 2, ch // 2, image=self.display_photo, anchor="center")

        self.stats_var.set(self._format_status())

        if elapsed_ms > 150:
            self._log(f"⏱️ Render took {elapsed_ms:.1f} ms")

    # ------------------------------------------------------------ detection
    def detect_craters(self):
        if self.processed_gray_f32 is None:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        try:
            t0 = time.perf_counter()
            self.detected_craters = self.engine.detect_craters(self.processed_gray_f32)
            self.craters_visible = True
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._log(f"🔴 Detected {len(self.detected_craters)} potential craters "
                     f"({elapsed_ms:.1f} ms)")
            self._on_change()
        except Exception as exc:
            messagebox.showerror("Crater detection failed", str(exc))
            self._log(f"❌ ERROR in crater detection: {exc}")

    def detect_sun_angle(self):
        if self.gray_f32 is None:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        try:
            craters = self.detected_craters
            if not craters:
                craters = self.engine.detect_craters(self.processed_gray_f32)
                self.detected_craters = craters
                self.craters_visible = True

            self.sun_angle, self.shadow_angle = self.engine.compute_sun_angle(
                self.gray_f32, craters
            )

            if np.isnan(self.sun_angle):
                self._log("☀️ Sun angle: could not be determined")
            else:
                self._log(f"☀️ Sun: {self.sun_angle:.0f}°, Shadow: {self.shadow_angle:.0f}°")

            self.stats_var.set(self._format_status())
            self._on_change()
        except Exception as exc:
            messagebox.showerror("Sun angle detection failed", str(exc))
            self._log(f"❌ ERROR in sun angle detection: {exc}")

    # ------------------------------------------------------------ misc UI
    def reset_all(self):
        if self.original_bgr is None:
            return
        self.contrast_var.set(DEFAULT_CONTRAST)
        self.threshold_var.set(DEFAULT_THRESHOLD)
        self.blur_var.set(DEFAULT_BLUR)
        self.method_var.set("Pencil")
        self.theme_var.set("Normal")
        self.grid_on_var.set(False)
        self.detected_craters = []
        self.craters_visible = False
        self.sun_angle = float("nan")
        self.shadow_angle = float("nan")
        self._log("🔄 Reset all settings")
        self._on_change()

    def download_image(self):
        if self.sketch_image is None:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), 
                      ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            self.sketch_image.save(file_path)
            messagebox.showinfo("Success", f"✅ Saved successfully!\n{file_path}")
            self._log(f"💾 Saved: {os.path.basename(file_path)}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            self._log(f"❌ ERROR saving image: {exc}")

    def _format_status(self):
        craters_str = f"{len(self.detected_craters)}" if self.craters_visible else "not run"
        if np.isnan(self.sun_angle):
            sun_str = "☀️ Sun: n/a, Shadow: n/a"
        else:
            sun_str = f"☀️ Sun: {self.sun_angle:.0f}°, Shadow: {self.shadow_angle:.0f}°"
        return f"🔴 Craters: {craters_str}   |   {sun_str}"

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')}  {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")


def main():
    root = tk.Tk()
    app = PlanetaryAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()