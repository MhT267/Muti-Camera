import sys
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QWidget, QVBoxLayout,
                             QHBoxLayout, QMainWindow, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import gxipy as gx
import PySpin
from ximea import xiapi

# === 通道配置 ===
camera_configs = {
    "42289050": {
        "index_map": [4, 3, 5, 7, 6, 8, 1, 0, 2],
        "wavelengths": [621, 638, 658, 677, 698, 719, 738, 757, 775]
    },
    "04283550": {
        "index_map": [4, 3, 5, 7, 6, 8, 1, 0, 2],
        "wavelengths": [453, 468, 484, 496, 513, 530, 540, 556, 570]
    }
}

WIDTH, HEIGHT = 2048, 2048

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Viewer")
        self.setGeometry(100, 100, 2448, 2048)

        self.init_daheng()
        self.init_flir()
        self.init_ximea("04283550", "ximea1")
        self.init_ximea("42289050", "ximea2")
        self.init_ui()

        self.daheng_thread = threading.Thread(target=self.update_daheng, daemon=True)
        self.flir_thread = threading.Thread(target=self.update_flir, daemon=True)
        self.ximea1_thread = threading.Thread(target=self.update_ximea, args=("ximea1",), daemon=True)
        self.ximea2_thread = threading.Thread(target=self.update_ximea, args=("ximea2",), daemon=True)

        self.daheng_thread.start()
        self.flir_thread.start()
        self.ximea1_thread.start()
        self.ximea2_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(33)

    def init_ui(self):
        layout = QGridLayout()
        self.daheng_label = QLabel("DAHENG")
        self.flir_label = QLabel("FLIR")
        self.ximea1_label = QLabel("XIMEA 04283550")
        self.ximea2_label = QLabel("XIMEA 42289050")
        for label in [self.daheng_label, self.flir_label, self.ximea1_label, self.ximea2_label]:
            label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.daheng_label, 0, 0)
        layout.addWidget(self.ximea1_label, 0, 1)
        layout.addWidget(self.flir_label, 1, 0)
        layout.addWidget(self.ximea2_label, 1, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def init_daheng(self):
        """Initialize Daheng camera (RGB + YOLO)"""
        self.manager = gx.DeviceManager()
        dev_num, dev_info_list = self.manager.update_device_list()
        if dev_num == 0:
            print("Daheng camera not connected")
            sys.exit(1)

        self.cam = self.manager.open_device_by_sn(dev_info_list[0].get("sn"))
        self.cam.stream_on()

    def init_flir(self):
        self.flir_system = PySpin.System.GetInstance()
        cam_list = self.flir_system.GetCameras()
        if cam_list.GetSize() == 0:
            print("No FLIR camera")
            sys.exit(1)
        self.flir_cam = cam_list[0]
        self.flir_cam.Init()
        self.flir_cam.BeginAcquisition()

    def init_ximea(self, sn, name):
        cam = xiapi.Camera()
        cam.open_device_by_SN(sn)
        cam.set_exposure(30000)
        cam.set_imgdataformat('XI_MONO8')
        cam.set_width(WIDTH)
        cam.set_height(HEIGHT)
        cam.start_acquisition()
        config = camera_configs[sn]
        setattr(self, f"{name}_cam", cam)
        setattr(self, f"{name}_index_map", config["index_map"])
        setattr(self, f"{name}_wavelengths", config["wavelengths"])

    def extract_raw_channels(self, mosaic_img):
        chs = []
        for dy in range(3):
            for dx in range(3):
                chs.append(mosaic_img[dy::3, dx::3])
        return chs

    def draw_mosaic_grid(self, channels, labels):
        grid_size = 682
        grid = np.zeros((2048, 2048), dtype=np.uint8)
        for idx, ch in enumerate(channels):
            row, col = divmod(idx, 3)
            y, x = row * grid_size, col * grid_size
            ch_resized = cv2.resize(ch, (grid_size, grid_size))
            grid[y:y+grid_size, x:x+grid_size] = ch_resized
            cv2.putText(grid, f"{labels[idx]}nm", (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        return grid

    def update_daheng(self):
        """Capture and update Daheng (RGB + YOLO) camera frames"""
        while True:
            raw = self.cam.data_stream[0].get_image()
            if raw is None:
                continue
            rgb = raw.convert("RGB").get_numpy_array()
            resized = cv2.resize(rgb, (1224, 1224))  # Example resolution
            self.daheng_frame = resized

    def update_flir(self):
        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        while True:
            image = self.flir_cam.GetNextImage(1000)
            if image.IsIncomplete():
                continue
            converted = processor.Convert(image, PySpin.PixelFormat_Mono8)
            array = np.frombuffer(converted.GetData(), dtype=np.uint8).reshape(converted.GetHeight(), converted.GetWidth())
            self.flir_frame = cv2.resize(array, (1024, 1024))
            image.Release()

    def update_ximea(self, name):
        cam = getattr(self, f"{name}_cam")
        index_map = getattr(self, f"{name}_index_map")
        wavelengths = getattr(self, f"{name}_wavelengths")
        while True:
            img = xiapi.Image()
            cam.get_image(img)
            raw = img.get_image_data_numpy()
            # 1. 提取通道
            chs_raw = self.extract_raw_channels_no_interp(raw)
            chs_sorted = [chs_raw[i] for i in index_map]
            # 2. 拼接图像（不插值）
            mosaic_small = self.draw_mosaic_grid_no_interp(chs_sorted, wavelengths)
            mosaic = cv2.resize(mosaic_small, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
            # 3. 翻转图像
            flipped = cv2.flip(mosaic, -1)
            # 4. 添加标签
            labeled = self.draw_labels_on_flipped_image(flipped, wavelengths)
            # 5. 存储图像
            setattr(self, f"{name}_frame", labeled)

    def extract_raw_channels_no_interp(self, mosaic_img):
        channels = []
        for dy in range(3):
            for dx in range(3):
                ch = mosaic_img[dy::3, dx::3]
                channels.append(ch)
        return channels

    def draw_mosaic_grid_no_interp(self, channels, labels):
        """将9个 682x682 的通道图像拼接为 2048x2048 画面（不做每通道插值）"""
        grid_size = 682  # 每个通道的大小是 682x682
        grid = np.zeros((2048, 2048), dtype=np.uint8)  # 创建空白拼接图像，大小为 2048x2048

        for idx, ch in enumerate(channels):  # 遍历所有通道
            row, col = divmod(idx, 3)  # 计算通道在网格中的位置
            y, x = row * grid_size, col * grid_size

            # 确保通道图像尺寸为 682x682
            ch_resized = cv2.resize(ch, (grid_size, grid_size))

            # 拼接图像
            grid[y:y + grid_size, x:x + grid_size] = ch_resized
            # cv2.putText(grid, f"{labels[idx]}nm", (x + 5, y + 25),  # 给每个通道添加波长标签
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        return grid

    def draw_labels_on_flipped_image(self, image, labels):
        """在翻转后的图像上添加标签（确保标签在左上角）"""
        grid_size = 682  # 每个通道的大小是 682x682
        for idx, label in enumerate(labels):  # 遍历所有标签
            row, col = divmod(idx, 3)  # 计算通道在网格中的位置
            y, x = row * grid_size, col * grid_size

            # 给图像添加标签，确保标签位置不受翻转影响
            cv2.putText(image, f"{label}nm", (x + 5, y + 25),  # 将标签放置在左上角
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        return image

    def set_pixmap(self, label, frame, gray=False):
        if frame is None:
            return
        if gray:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            rgb = frame
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(1024, 1024, Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def update_gui(self):
        self.set_pixmap(self.daheng_label, getattr(self, "daheng_frame", None))
        self.set_pixmap(self.flir_label, getattr(self, "flir_frame", None), gray=True)
        self.set_pixmap(self.ximea1_label, getattr(self, "ximea1_frame", None), gray=True)
        self.set_pixmap(self.ximea2_label, getattr(self, "ximea2_frame", None), gray=True)

    def closeEvent(self, event):
        self.exit_flag = True
        time.sleep(0.5)
        self.cam.stream_off()
        self.cam.close_device()
        self.flir_cam.EndAcquisition()
        self.flir_cam.DeInit()
        self.flir_system.ReleaseInstance()
        self.ximea1_cam.stop_acquisition()
        self.ximea1_cam.close_device()
        self.ximea2_cam.stop_acquisition()
        self.ximea2_cam.close_device()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = CameraApp()
    viewer.show()
    sys.exit(app.exec_())
