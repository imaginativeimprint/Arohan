import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel, QSlider, QGroupBox, QFormLayout
)

class ImageProcessorThread(QThread):
    """Worker thread to run OpenCV processing without blocking the UI main loop."""
    processed_signal = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.image = None
        self.ksize = 21
        self.sigma = 0
        self.shade_scale = 1.0
        self.sharpness = 0.0

    def update_params(self, image, ksize, sigma, shade_scale, sharpness):
        self.image = image
        # Kernel size for Gaussian blur must be an odd integer
        self.ksize = ksize if ksize % 2 != 0 else ksize + 1
        self.sigma = sigma
        self.shade_scale = shade_scale
        self.sharpness = sharpness
        if not self.isRunning():
            self.start()

    def run(self):
        if self.image is None:
            return

        # 1. Convert image to Grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 2. Invert Grayscale image
        inverted = cv2.bitwise_not(gray)

        # 3. Apply Gaussian Blur to inverted image
        blurred = cv2.GaussianBlur(inverted, (self.ksize, self.ksize), self.sigma)

        # 4. Invert blurred image
        inverted_blur = cv2.bitwise_not(blurred)

        # 5. Pencil sketch effect using Color Dodge division
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)

        # 6. Adjust shade intensity/contrast
        if self.shade_scale != 1.0:
            sketch = cv2.multiply(sketch, self.shade_scale)
            sketch = np.clip(sketch, 0, 255).astype(np.uint8)

        # 7. Apply Sharpness / Detailing (Unsharp Masking)
        if self.sharpness > 0.0:
            sketch_blur = cv2.GaussianBlur(sketch, (3, 3), 0)
            sketch = cv2.addWeighted(sketch, 1.0 + self.sharpness, sketch_blur, -self.sharpness, 0)
            sketch = np.clip(sketch, 0, 255).astype(np.uint8)

        self.processed_signal.emit(sketch)


class PencilSketchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Pencil Sketch Studio")
        self.resize(1100, 700)

        self.original_cv_img = None
        self.current_sketch = None

        # Threading for real-time responsiveness
        self.processor_thread = ImageProcessorThread()
        self.processor_thread.processed_signal.connect(self.display_processed_image)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Column: Image Preview Canvas
        self.image_label = QLabel("Click 'Load Image' to start")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #666; background-color: #1e1e1e; color: #aaa; font-size: 16px;")
        
        # Prevent layout resizing loop
        self.image_label.setMinimumSize(1, 1)
        
        main_layout.addWidget(self.image_label, stretch=3)

        # Right Column: Controls Sidebar
        sidebar = QVBoxLayout()
        main_layout.addLayout(sidebar, stretch=1)

        # Buttons
        self.btn_load = QPushButton("Load Image")
        self.btn_load.setFixedHeight(40)
        self.btn_load.clicked.connect(self.load_image)
        sidebar.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save Sketch")
        self.btn_save.setFixedHeight(40)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_image)
        sidebar.addWidget(self.btn_save)

        # Parameters Group
        param_group = QGroupBox("Sketch Parameters")
        form_layout = QFormLayout()

        # 1. Blur Kernel Size Slider (Stroke Softness)
        self.slider_ksize = QSlider(Qt.Horizontal)
        self.slider_ksize.setRange(3, 99)
        self.slider_ksize.setValue(21)
        self.slider_ksize.setSingleStep(2)
        self.lbl_ksize = QLabel("21")
        self.slider_ksize.valueChanged.connect(self.on_param_change)
        form_layout.addRow("Blur Kernel Size:", self.slider_ksize)
        form_layout.addRow("", self.lbl_ksize)

        # 2. Gaussian Sigma Slider (Edge Smoothness)
        self.slider_sigma = QSlider(Qt.Horizontal)
        self.slider_sigma.setRange(0, 50)
        self.slider_sigma.setValue(0)
        self.lbl_sigma = QLabel("0 (Auto)")
        self.slider_sigma.valueChanged.connect(self.on_param_change)
        form_layout.addRow("Sigma (Smoothness):", self.slider_sigma)
        form_layout.addRow("", self.lbl_sigma)

        # 3. Shading / Darkness Intensity Slider
        self.slider_shade = QSlider(Qt.Horizontal)
        self.slider_shade.setRange(10, 200)
        self.slider_shade.setValue(100)
        self.lbl_shade = QLabel("1.0x")
        self.slider_shade.valueChanged.connect(self.on_param_change)
        form_layout.addRow("Shade Intensity:", self.slider_shade)
        form_layout.addRow("", self.lbl_shade)

        # 4. Sharpness / Detail Slider
        self.slider_sharpness = QSlider(Qt.Horizontal)
        self.slider_sharpness.setRange(0, 30)
        self.slider_sharpness.setValue(0)
        self.lbl_sharpness = QLabel("0.0")
        self.slider_sharpness.valueChanged.connect(self.on_param_change)
        form_layout.addRow("Detail Sharpness:", self.slider_sharpness)
        form_layout.addRow("", self.lbl_sharpness)

        param_group.setLayout(form_layout)
        sidebar.addWidget(param_group)
        sidebar.addStretch()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.webp)")
        if file_path:
            self.original_cv_img = cv2.imread(file_path)
            self.btn_save.setEnabled(True)
            self.trigger_processing()

    def on_param_change(self):
        # Update UI Labels
        kval = self.slider_ksize.value()
        if kval % 2 == 0:
            kval += 1
        self.lbl_ksize.setText(str(kval))

        sigval = self.slider_sigma.value()
        self.lbl_sigma.setText(str(sigval) if sigval > 0 else "0 (Auto)")

        shadeval = self.slider_shade.value() / 100.0
        self.lbl_shade.setText(f"{shadeval:.2f}x")

        sharpval = self.slider_sharpness.value() / 10.0
        self.lbl_sharpness.setText(f"{sharpval:.1f}")

        self.trigger_processing()

    def trigger_processing(self):
        if self.original_cv_img is None:
            return

        ksize = self.slider_ksize.value()
        sigma = self.slider_sigma.value()
        shade_scale = self.slider_shade.value() / 100.0
        sharpness = self.slider_sharpness.value() / 10.0

        self.processor_thread.update_params(self.original_cv_img, ksize, sigma, shade_scale, sharpness)

    @Slot(np.ndarray)
    def display_processed_image(self, sketch_img):
        self.current_sketch = sketch_img
        h, w = sketch_img.shape
        bytes_per_line = w
        q_img = QImage(sketch_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)
        
        target_size = self.image_label.contentsRect().size()
        if not target_size.isEmpty():
            scaled_pixmap = pixmap.scaled(
                target_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def save_image(self):
        if self.current_sketch is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sketch Image", "sketch_output.png", "PNG (*.png);;JPEG (*.jpg)")
        if file_path:
            cv2.imwrite(file_path, self.current_sketch)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_sketch is not None:
            self.display_processed_image(self.current_sketch)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PencilSketchApp()
    window.show()
    sys.exit(app.exec())