import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2
import os

class ImageSketchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image to Sketch/Edge Detection Converter - Pro")
        self.root.geometry("1200x750")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.original_image = None
        self.sketch_image = None
        self.processed_image = None
        self.original_photo = None
        self.sketch_photo = None
        self.edge_method = tk.StringVar(value="pencil")
        
        # Zoom and contrast variables (Setting to your perfect values)
        self.original_zoom = 1.0
        self.sketch_zoom = 1.0
        self.contrast_value = 2.2
        self.threshold_value = 15
        self.blur_value = 10
        
        # Original image for processing
        self.original_display_image = None
        self.sketch_display_image = None
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="🎨 Professional Image to Sketch & Crater Detector", 
            font=('Arial', 22, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)
        
        # Top Control Panel
        top_control = tk.Frame(self.root, bg='#2c3e50')
        top_control.pack(pady=10, fill=tk.X)
        
        # Upload Button
        upload_btn = tk.Button(
            top_control,
            text="📁 Upload Image",
            command=self.upload_image,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=8,
            relief=tk.RAISED,
            borderwidth=2
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Edge Detection Method Selection
        tk.Label(
            top_control,
            text="Method:",
            font=('Arial', 12, 'bold'),
            bg='#2c3e50',
            fg='white'
        ).pack(side=tk.LEFT, padx=(20, 5))
        
        methods = [
            ("Sobel", "sobel"),
            ("Laplacian", "laplacian"),
            ("Canny", "canny"),
            ("Pencil (Shaded)", "pencil")
        ]
        
        for text, value in methods:
            tk.Radiobutton(
                top_control,
                text=text,
                variable=self.edge_method,
                value=value,
                bg='#2c3e50',
                fg='white',
                selectcolor='#2c3e50',
                font=('Arial', 10),
                command=self.apply_edge_detection
            ).pack(side=tk.LEFT, padx=5)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Original Image
        left_frame = tk.Frame(main_frame, bg='#2c3e50')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        original_header = tk.Frame(left_frame, bg='#34495e')
        original_header.pack(fill=tk.X)
        
        tk.Label(
            original_header,
            text="📷 Original Image",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Original zoom controls
        zoom_frame_orig = tk.Frame(original_header, bg='#34495e')
        zoom_frame_orig.pack(side=tk.RIGHT, padx=5)
        
        tk.Button(zoom_frame_orig, text="➕", command=lambda: self.zoom_original(1.2), bg='#2c3e50', fg='white', font=('Arial', 12, 'bold'), padx=5).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame_orig, text="➖", command=lambda: self.zoom_original(0.8), bg='#2c3e50', fg='white', font=('Arial', 12, 'bold'), padx=5).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame_orig, text="1:1", command=lambda: self.reset_zoom_original(), bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'), padx=5).pack(side=tk.LEFT, padx=2)
        
        self.original_canvas_frame = tk.Frame(left_frame, bg='#34495e', relief=tk.GROOVE, borderwidth=3)
        self.original_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.original_canvas = tk.Canvas(self.original_canvas_frame, bg='#2c3e50', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side - Sketch Image
        right_frame = tk.Frame(main_frame, bg='#2c3e50')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        sketch_header = tk.Frame(right_frame, bg='#34495e')
        sketch_header.pack(fill=tk.X)
        
        tk.Label(
            sketch_header,
            text="✏️ Sketch Output",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Sketch zoom controls
        zoom_frame_sketch = tk.Frame(sketch_header, bg='#34495e')
        zoom_frame_sketch.pack(side=tk.RIGHT, padx=5)
        
        tk.Button(zoom_frame_sketch, text="➕", command=lambda: self.zoom_sketch(1.2), bg='#2c3e50', fg='white', font=('Arial', 12, 'bold'), padx=5).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame_sketch, text="➖", command=lambda: self.zoom_sketch(0.8), bg='#2c3e50', fg='white', font=('Arial', 12, 'bold'), padx=5).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame_sketch, text="1:1", command=lambda: self.reset_zoom_sketch(), bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'), padx=5).pack(side=tk.LEFT, padx=2)
        
        self.sketch_canvas_frame = tk.Frame(right_frame, bg='#34495e', relief=tk.GROOVE, borderwidth=3)
        self.sketch_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.sketch_canvas = tk.Canvas(self.sketch_canvas_frame, bg='#2c3e50', highlightthickness=0)
        self.sketch_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom Control Panel - Editing Tools
        bottom_control = tk.Frame(self.root, bg='#2c3e50')
        bottom_control.pack(pady=10, fill=tk.X, padx=10)
        
        # Contrast Control
        contrast_frame = tk.Frame(bottom_control, bg='#2c3e50')
        contrast_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(contrast_frame, text="Contrast:", font=('Arial', 11, 'bold'), bg='#2c3e50', fg='white').pack(side=tk.LEFT, padx=5)
        
        self.contrast_slider = tk.Scale(contrast_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, length=150, bg='#2c3e50', fg='white', highlightthickness=0, command=self.update_contrast)
        self.contrast_slider.set(2.2)
        self.contrast_slider.pack(side=tk.LEFT, padx=5)
        
        self.contrast_label = tk.Label(contrast_frame, text="2.2x", font=('Arial', 10), bg='#2c3e50', fg='#ecf0f1', width=5)
        self.contrast_label.pack(side=tk.LEFT)
        
        # Threshold Control
        threshold_frame = tk.Frame(bottom_control, bg='#2c3e50')
        threshold_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(threshold_frame, text="Threshold:", font=('Arial', 11, 'bold'), bg='#2c3e50', fg='white').pack(side=tk.LEFT, padx=5)
        
        self.threshold_slider = tk.Scale(threshold_frame, from_=0, to=200, resolution=5, orient=tk.HORIZONTAL, length=150, bg='#2c3e50', fg='white', highlightthickness=0, command=self.update_threshold)
        self.threshold_slider.set(15)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)
        
        self.threshold_label = tk.Label(threshold_frame, text="15", font=('Arial', 10), bg='#2c3e50', fg='#ecf0f1', width=5)
        self.threshold_label.pack(side=tk.LEFT)
        
        # Blur Control
        blur_frame = tk.Frame(bottom_control, bg='#2c3e50')
        blur_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(blur_frame, text="Blur:", font=('Arial', 11, 'bold'), bg='#2c3e50', fg='white').pack(side=tk.LEFT, padx=5)
        
        self.blur_slider = tk.Scale(blur_frame, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL, length=150, bg='#2c3e50', fg='white', highlightthickness=0, command=self.update_blur)
        self.blur_slider.set(10)
        self.blur_slider.pack(side=tk.LEFT, padx=5)
        
        self.blur_label = tk.Label(blur_frame, text="10", font=('Arial', 10), bg='#2c3e50', fg='#ecf0f1', width=5)
        self.blur_label.pack(side=tk.LEFT)
        
        # Action Buttons
        action_frame = tk.Frame(bottom_control, bg='#2c3e50')
        action_frame.pack(side=tk.RIGHT, padx=20)
        
        # Detect Craters Button
        detect_btn = tk.Button(
            action_frame,
            text="🔵 Detect Craters",
            command=self.detect_craters,
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2
        )
        detect_btn.pack(side=tk.LEFT, padx=5)
        
        # Invert Button
        invert_btn = tk.Button(
            action_frame,
            text="🔄 Invert",
            command=self.invert_sketch,
            font=('Arial', 11, 'bold'),
            bg='#e67e22',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2
        )
        invert_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset Button
        reset_btn = tk.Button(
            action_frame,
            text="🔄 Reset",
            command=self.reset_all,
            font=('Arial', 11, 'bold'),
            bg='#95a5a6',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Download Button
        download_btn = tk.Button(
            action_frame,
            text="💾 Download",
            command=self.download_image,
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2
        )
        download_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="📌 Upload an image to convert to sketch",
            font=('Arial', 11),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.status_label.pack(pady=5)
        
        # Bind window resize
        self.root.bind('<Configure>', self.on_window_resize)
        
    def on_window_resize(self, event):
        if self.original_image:
            self.display_original_image()
        if self.sketch_image:
            self.display_sketch_image()
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.original_image = Image.open(file_path)
            self.original_display_image = self.original_image.copy()
            
            # Reset controls to perfect preset
            self.contrast_slider.set(2.2)
            self.threshold_slider.set(15)
            self.blur_slider.set(10)
            self.original_zoom = 1.0
            self.sketch_zoom = 1.0
            self.edge_method.set("pencil")
            
            self.apply_edge_detection()
            
            self.status_label.config(text=f"✅ Image loaded: {os.path.basename(file_path)}", fg='#2ecc71')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_label.config(text="❌ Error loading image", fg='#e74c3c')
    
    def apply_edge_detection(self):
        if self.original_image is None:
            return
        
        method = self.edge_method.get()
        
        try:
            gray = self.original_image.convert('L')
            
            if method == "sobel":
                self.sketch_image = gray.filter(ImageFilter.FIND_EDGES)
                self.sketch_image = ImageOps.invert(self.sketch_image)
                enhancer = ImageEnhance.Contrast(self.sketch_image)
                self.sketch_image = enhancer.enhance(self.contrast_value)
                
            elif method == "laplacian":
                self.sketch_image = gray.filter(ImageFilter.Kernel((3,3), (-1,-1,-1,-1,8,-1,-1,-1,-1), scale=1, offset=0))
                self.sketch_image = ImageOps.invert(self.sketch_image)
                self.sketch_image = self.sketch_image.filter(ImageFilter.EDGE_ENHANCE)
                enhancer = ImageEnhance.Contrast(self.sketch_image)
                self.sketch_image = enhancer.enhance(self.contrast_value)
                
            elif method == "canny":
                blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
                edges = blurred.filter(ImageFilter.FIND_EDGES)
                self.sketch_image = self.apply_threshold(edges, self.threshold_value)
                self.sketch_image = ImageOps.invert(self.sketch_image)
                enhancer = ImageEnhance.Contrast(self.sketch_image)
                self.sketch_image = enhancer.enhance(self.contrast_value)
                
            elif method == "pencil":
                # Base Pencil Sketch
                inverted = ImageOps.invert(gray)
                blurred = inverted.filter(ImageFilter.GaussianBlur(radius=self.blur_value))
                sketch = self.dodge_images(gray, blurred)
                enhancer = ImageEnhance.Contrast(sketch)
                sketch = enhancer.enhance(self.contrast_value)
                
                # Advanced Shading for 3D Craters
                sketch = self.apply_professional_shading(sketch)
                
                self.sketch_image = sketch
            
            self.processed_image = self.sketch_image.copy()
            self.display_original_image()
            self.display_sketch_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Edge detection failed:\n{str(e)}")
    
    def apply_professional_shading(self, pil_image):
        try:
            open_cv_image = np.array(pil_image)
            
            # Bilateral Filter: Smooths while keeping edges sharp
            bilateral = cv2.bilateralFilter(open_cv_image, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Emboss Kernel: Creates 3D light/shadow effect
            kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
            embossed = cv2.filter2D(bilateral, -1, kernel)
            
            # Blend for natural shading
            final_shaded = cv2.addWeighted(bilateral, 0.7, embossed, 0.3, 0)
            
            # Normalize to pop contrast
            final_shaded = cv2.normalize(final_shaded, None, 0, 255, cv2.NORM_MINMAX)
            
            return Image.fromarray(final_shaded)
        except Exception as e:
            print(f"Error in advanced shading: {e}")
            return pil_image
    
    def detect_craters(self):
        """Heavy-Duty Detection for noisy, low-contrast surfaces."""
        if self.processed_image is None:
            messagebox.showwarning("No Image", "Please generate a sketch first!")
            return

        try:
            # Convert PIL sketch to OpenCV format
            open_cv_image = np.array(self.processed_image.convert('L'))

            # 1. STRONG NOISE REDUCTION
            blurred = cv2.GaussianBlur(open_cv_image, (15, 15), 0)

            # 2. CONTRAST ENHANCEMENT (Clahe)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)

            # 3. THRESHOLDING (Otsu's method)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 4. MORPHOLOGICAL CLEANUP
            kernel = np.ones((5, 5), np.uint8)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)

            # 5. FIND CONTOURS
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 6. SMARTER FILTERING (Circularity Check)
            colored_sketch = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
            crater_count = 0

            for c in contours:
                area = cv2.contourArea(c)
                # Filter by size (Adjust 200 and 50000 based on crater size)
                if 200 < area < 50000:  
                    perimeter = cv2.arcLength(c, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        # Crater check: Must be somewhat circular
                        if 0.1 < circularity < 1.0:
                            cv2.drawContours(colored_sketch, [c], -1, (255, 0, 0), 2) # Blue outline
                            crater_count += 1

            # Convert back to PIL
            detected_image = Image.fromarray(colored_sketch)
            
            # Update the displayed sketch
            self.sketch_image = detected_image
            self.processed_image = detected_image.copy()
            self.display_sketch_image()
            
            self.status_label.config(text=f"🔵 Detected {crater_count} potential craters", fg='#3498db')

        except Exception as e:
            messagebox.showerror("Error", f"Crater detection failed:\n{str(e)}")
    
    def apply_threshold(self, image, threshold_value):
        arr = np.array(image)
        arr = np.where(arr > threshold_value, 255, 0)
        return Image.fromarray(arr.astype('uint8'))
    
    def dodge_images(self, base, blend):
        base_arr = np.array(base, dtype=np.float32)
        blend_arr = np.array(blend, dtype=np.float32)
        result = (base_arr * 255) / (255 - blend_arr + 1)
        result = np.clip(result, 0, 255)
        return Image.fromarray(result.astype('uint8'))
    
    # --- FIXED ZOOM LOGIC ---
    def display_original_image(self):
        if self.original_image is None:
            return
        
        canvas_width = max(self.original_canvas.winfo_width(), 400)
        canvas_height = max(self.original_canvas.winfo_height(), 400)
        
        display_image = self.original_image.copy()
        width, height = display_image.size
        new_width = int(width * self.original_zoom)
        new_height = int(height * self.original_zoom)
        
        if new_width > canvas_width or new_height > canvas_height:
            aspect = new_width / new_height
            if canvas_width / canvas_height > aspect:
                display_width = int(canvas_height * aspect)
                display_height = canvas_height
            else:
                display_width = canvas_width
                display_height = int(canvas_width / aspect)
        else:
            display_width = new_width
            display_height = new_height
        
        display_image = display_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.original_photo = ImageTk.PhotoImage(display_image)
        
        self.original_canvas.delete("all")
        x = max((canvas_width - display_width) // 2, 0)
        y = max((canvas_height - display_height) // 2, 0)
        
        self.original_canvas.create_image(x, y, image=self.original_photo, anchor=tk.NW)
        info_text = f"{self.original_image.width}×{self.original_image.height} | Zoom: {self.original_zoom:.1f}x"
        self.original_canvas.create_text(10, 10, text=info_text, anchor=tk.NW, fill='white', font=('Arial', 10, 'bold'))
    
    def display_sketch_image(self):
        if self.sketch_image is None:
            return
        
        canvas_width = max(self.sketch_canvas.winfo_width(), 400)
        canvas_height = max(self.sketch_canvas.winfo_height(), 400)
        
        display_image = self.sketch_image.copy()
        width, height = display_image.size
        new_width = int(width * self.sketch_zoom)
        new_height = int(height * self.sketch_zoom)
        
        if new_width > canvas_width or new_height > canvas_height:
            aspect = new_width / new_height
            if canvas_width / canvas_height > aspect:
                display_width = int(canvas_height * aspect)
                display_height = canvas_height
            else:
                display_width = canvas_width
                display_height = int(canvas_width / aspect)
        else:
            display_width = new_width
            display_height = new_height
        
        display_image = display_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.sketch_photo = ImageTk.PhotoImage(display_image)
        
        self.sketch_canvas.delete("all")
        x = max((canvas_width - display_width) // 2, 0)
        y = max((canvas_height - display_height) // 2, 0)
        
        self.sketch_canvas.create_image(x, y, image=self.sketch_photo, anchor=tk.NW)
        info_text = f"Method: {self.edge_method.get().title()} | Zoom: {self.sketch_zoom:.1f}x"
        self.sketch_canvas.create_text(10, 10, text=info_text, anchor=tk.NW, fill='white', font=('Arial', 10, 'bold'))
    
    def zoom_original(self, factor):
        self.original_zoom *= factor
        self.original_zoom = max(0.1, min(5.0, self.original_zoom))
        self.display_original_image()
    
    def reset_zoom_original(self):
        self.original_zoom = 1.0
        self.display_original_image()
    
    def zoom_sketch(self, factor):
        self.sketch_zoom *= factor
        self.sketch_zoom = max(0.1, min(5.0, self.sketch_zoom))
        self.display_sketch_image()
    
    def reset_zoom_sketch(self):
        self.sketch_zoom = 1.0
        self.display_sketch_image()
    
    def update_contrast(self, value):
        self.contrast_value = float(value)
        self.contrast_label.config(text=f"{self.contrast_value:.1f}x")
        self.apply_edge_detection()
    
    def update_threshold(self, value):
        self.threshold_value = int(value)
        self.threshold_label.config(text=str(self.threshold_value))
        if self.edge_method.get() in ["canny", "pencil"]:
            self.apply_edge_detection()
    
    def update_blur(self, value):
        self.blur_value = int(value)
        self.blur_label.config(text=str(self.blur_value))
        if self.edge_method.get() == "pencil":
            self.apply_edge_detection()
    
    def invert_sketch(self):
        if self.sketch_image is None:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        
        self.sketch_image = ImageOps.invert(self.sketch_image)
        self.processed_image = self.sketch_image.copy()
        self.display_sketch_image()
        self.status_label.config(text="🔄 Colors inverted", fg='#f1c40f')
    
    def reset_all(self):
        if self.original_image is None:
            return
        
        self.contrast_slider.set(2.2)
        self.threshold_slider.set(15)
        self.blur_slider.set(10)
        self.original_zoom = 1.0
        self.sketch_zoom = 1.0
        self.contrast_value = 2.2
        self.threshold_value = 15
        self.blur_value = 10
        self.edge_method.set("pencil")
        
        self.apply_edge_detection()
        self.status_label.config(text="🔄 Reset all settings", fg='#f1c40f')
    
    def download_image(self):
        if self.sketch_image is None:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Sketch Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.sketch_image.save(file_path)
            messagebox.showinfo("Success", f"✅ Sketch saved successfully!\n{file_path}")
            self.status_label.config(text=f"💾 Saved: {os.path.basename(file_path)}", fg='#2ecc71')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
            self.status_label.config(text="❌ Error saving image", fg='#e74c3c')

def main():
    root = tk.Tk()
    app = ImageSketchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()