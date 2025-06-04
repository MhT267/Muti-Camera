from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton,
                             QWidget, QHBoxLayout, QVBoxLayout,
                             QMainWindow, QStatusBar, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np
import threading
import gxipy as gx
import PySpin
from ximea import xiapi
import sys
import time
from datetime import datetime
from ultralytics import YOLO

# 初始化YOLO模型
model = YOLO("./checkpoints/yolov8x.pt")


# === 相机图像参数 ===
WIDTH = 2048  # 图像的宽度
HEIGHT = 2048  # 图像的高度

# === 通道映射与波长配置 ===
camera_configs = {
    "42289050": {  # 针对不同设备 SN 的配置
        "index_map": [4, 3, 5, 7, 6, 8, 1, 0, 2],  # 通道映射，调整为与波长对应
        "wavelengths": [621, 638, 658, 677, 698, 719, 738, 757, 775]  # 每个通道对应的波长（单位：nm）
    },
    "04283550": {
        "index_map": [4, 3, 5, 7, 6, 8, 1, 0, 2],
        "wavelengths": [453, 468, 484, 496, 513, 530, 540, 556, 570]
    }
}


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Three Camera Display with YOLO")
        self.setGeometry(100, 100, 1224 * 3, 1024)  # 适合三屏并列显示

        # 共享数据
        self.daheng_frame = None
        self.annotated_frame = None
        self.flir_frame = None
        self.ximea_frame = None
        self.lock = threading.Lock()
        self.exit_flag = False
        self.fps = 0

        # 初始化界面
        self.init_ui()

        # 初始化相机
        self.init_daheng()
        self.init_flir()
        self.init_ximea()

        # 启动线程
        self.daheng_thread = threading.Thread(target=self.update_daheng)
        self.infer_thread = threading.Thread(target=self.inference)
        self.flir_thread = threading.Thread(target=self.update_flir)
        self.ximea_thread = threading.Thread(target=self.update_ximea)

        for t in [self.daheng_thread, self.infer_thread, self.flir_thread, self.ximea_thread]:
            t.daemon = True
            t.start()

        # 定时器刷新界面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)  # 约33fps

    def init_ui(self):
        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 大恒相机区域 (RGB + YOLO)
        daheng_layout = QVBoxLayout()
        daheng_layout.setSpacing(0)
        self.daheng_title = QLabel("RGB + YOLO")
        self.daheng_title.setFixedHeight(40)
        self.daheng_title.setAlignment(Qt.AlignCenter)
        daheng_layout.addWidget(self.daheng_title, 0, Qt.AlignCenter)
        self.daheng_label = QLabel()
        self.daheng_label.setAlignment(Qt.AlignCenter)
        daheng_layout.addWidget(self.daheng_label)

        # FLIR相机区域
        flir_layout = QVBoxLayout()
        flir_layout.setSpacing(0)
        self.flir_title = QLabel("POL")
        self.flir_title.setFixedHeight(40)
        self.flir_title.setAlignment(Qt.AlignCenter)
        flir_layout.addWidget(self.flir_title, 0, Qt.AlignCenter)
        self.flir_label = QLabel()
        self.flir_label.setAlignment(Qt.AlignCenter)
        flir_layout.addWidget(self.flir_label)

        # XIMEA相机区域
        ximea_layout = QVBoxLayout()
        ximea_layout.setSpacing(0)
        self.ximea_title = QLabel("XIMEA")
        self.ximea_title.setFixedHeight(40)
        self.ximea_title.setAlignment(Qt.AlignCenter)
        ximea_layout.addWidget(self.ximea_title, 0, Qt.AlignCenter)
        self.ximea_label = QLabel()
        self.ximea_label.setAlignment(Qt.AlignCenter)
        ximea_layout.addWidget(self.ximea_label)

        # 合并主布局
        main_layout.addLayout(daheng_layout)
        main_layout.addLayout(flir_layout)
        main_layout.addLayout(ximea_layout)

        # 控制按钮区域
        self.fps_label = QLabel("FPS: 0.0")
        self.save_button = QPushButton("保存截图")
        self.exit_button = QPushButton("退出")

        self.save_button.clicked.connect(self.save_snapshot)
        self.exit_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.fps_label)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.exit_button)

        # 整体布局
        container = QVBoxLayout()
        container.addLayout(main_layout)
        container.addLayout(button_layout)

        central_widget = QWidget()
        central_widget.setLayout(container)
        self.setCentralWidget(central_widget)

        # 状态栏
        self.statusBar().setStyleSheet("font: 12pt;")

    def init_daheng(self):
        """Initialize Daheng camera (RGB + YOLO)"""
        try:
            self.manager = gx.DeviceManager()
            dev_num, dev_info_list = self.manager.update_device_list()
            if dev_num == 0:
                raise RuntimeError("大恒相机未连接")
            self.cam = self.manager.open_device_by_sn(dev_info_list[0].get("sn"))
            self.cam.stream_on()
        except Exception as e:
            QMessageBox.critical(self, "相机错误", f"大恒相机初始化失败: {str(e)}")
            sys.exit(1)

    def init_flir(self):
        """Initialize FLIR camera (POL)"""
        try:
            self.flir_system = PySpin.System.GetInstance()
            cam_list = self.flir_system.GetCameras()
            if cam_list.GetSize() == 0:
                raise RuntimeError("未检测到FLIR相机")
            self.flir_cam = cam_list[0]
            self.flir_cam.Init()
            self.flir_cam.BeginAcquisition()
        except Exception as e:
            QMessageBox.critical(self, "相机错误", f"FLIR相机初始化失败: {str(e)}")
            sys.exit(1)

    def init_ximea(self):
        """Initialize XIMEA camera"""
        try:
            self.ximea_cam = xiapi.Camera()
            self.ximea_cam.open_device()
            self.ximea_cam.set_exposure(30000)
            self.ximea_cam.set_imgdataformat('XI_MONO8')
            self.ximea_cam.set_width(2048)
            self.ximea_cam.set_height(2048)
            self.ximea_cam.start_acquisition()

            # 获取设备SN号并选择配置
            device_sens_sn = self.ximea_cam.get_device_sn(buffer_size=256).decode('utf-8')
            camera_config = camera_configs.get(device_sens_sn, None)

            if camera_config is None:
                raise ValueError(f"无法找到匹配的设备配置，设备 SN: {device_sens_sn}")

            self.index_map = camera_config["index_map"]
            self.wavelengths = camera_config["wavelengths"]
        except Exception as e:
            QMessageBox.critical(self, "相机错误", f"XIMEA相机初始化失败: {str(e)}")
            sys.exit(1)

    def update_daheng(self):
        """Capture and update Daheng (RGB + YOLO) camera frames"""
        while not self.exit_flag:
            try:
                raw = self.cam.data_stream[0].get_image()
                if raw is None:
                    continue
                rgb = raw.convert("RGB").get_numpy_array()
                resized = cv2.resize(rgb, (1224, 720))
                with self.lock:
                    self.daheng_frame = resized.copy()
            except Exception as e:
                print(f"大恒相机更新错误: {e}")

    # def inference(self):
    #     """Perform YOLO inference on Daheng frames"""
    #     last_time = time.time()
    #     while not self.exit_flag:
    #         with self.lock:
    #             if self.daheng_frame is None:
    #                 continue
    #             img = self.daheng_frame.copy()
    #
    #         # 推理
    #         results = model.predict(img, imgsz=416, device='0',
    #                                 half=True, conf=0.3, verbose=False, max_det=10)
    #         self.annotated_frame = results[0].plot(
    #             conf=True, boxes=True, labels=True, kpt_line=True)
    #         now = time.time()
    #         self.fps = 1.0 / (now - last_time)
    #         last_time = now

    def inference(self):
        # 预热一次模型，避免首次推理卡顿
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=640, device='0', half=True, verbose=False)

        last_time = time.time()
        while not self.exit_flag:
            with self.lock:
                if self.daheng_frame is None:
                    continue
                original_frame = self.daheng_frame.copy()

            # Resize for YOLOv8 inference
            resized_frame = cv2.resize(original_frame, (640, 640))
            results = model.predict(resized_frame, imgsz=640, device='0',
                                    half=True, stream=True, verbose=False)

            for r in results:
                annotated_small = r.plot()
                annotated_full = cv2.resize(annotated_small, (original_frame.shape[1], original_frame.shape[0]))
                with self.lock:
                    self.annotated_frame = annotated_full

            now = time.time()
            self.fps = 1.0 / (now - last_time)
            last_time = now

    def update_flir(self):
        """Capture and update FLIR camera frames"""
        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        while not self.exit_flag:
            try:
                image = self.flir_cam.GetNextImage(1000)
                if image.IsIncomplete():
                    continue
                converted = processor.Convert(image, PySpin.PixelFormat_Mono8)
                array = np.frombuffer(converted.GetData(), dtype=np.uint8).reshape(
                    converted.GetHeight(), converted.GetWidth())
                resized = cv2.resize(array, (1224, 720))
                color_mono = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                with self.lock:
                    self.flir_frame = color_mono
                image.Release()
            except Exception as e:
                print(f"FLIR相机更新错误: {e}")

    def update_ximea(self):
        """Capture and update XIMEA camera frames"""
        while not self.exit_flag:
            try:
                img = xiapi.Image()
                self.ximea_cam.get_image(img)
                ximea_raw = img.get_image_data_numpy()

                # 提取和排序每个通道
                chs_raw = self.extract_raw_channels_no_interp(ximea_raw)
                chs_sorted = [chs_raw[i] for i in self.index_map]

                # 拼接通道图像并添加标签
                mosaic_small = self.draw_mosaic_grid_no_interp(chs_sorted, self.wavelengths)
                mosaic = cv2.resize(mosaic_small, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

                # 进行翻转操作（上下翻转）
                flipped_ximea = cv2.flip(mosaic, -1)  # 上下翻转图像

                # 重新定位标签
                flipped_ximea_with_labels = self.draw_labels_on_flipped_image(flipped_ximea, self.wavelengths)

                with self.lock:
                    self.ximea_frame = flipped_ximea_with_labels
            except Exception as e:
                print(f"XIMEA相机更新错误: {e}")

    def extract_raw_channels_no_interp(self, mosaic_img):
        """仅提取 9 通道，不插值，每个通道 340x340"""
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

    def set_pixmap(self, label, frame):
        """Convert OpenCV image to QPixmap and update the QLabel"""
        if frame is None:
            return

        if len(frame.shape) == 2:  # 如果 frame 是灰度图，rgb.shape 只有两个维度（H, W），取 shape[:2] 没问题，但 rgb.data 是 1 通道，交给 Format_RGB888 就会非法读取内存，导致崩溃
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            rgb = frame

        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)


    def update_gui(self):
        with self.lock:
            # Update Daheng camera display (with YOLO annotations)
            daheng_display = self.annotated_frame if self.annotated_frame is not None else self.daheng_frame
            if daheng_display is not None:
                self.set_pixmap(self.daheng_label, daheng_display)

            # Update FLIR camera display
            if self.flir_frame is not None:
                self.set_pixmap(self.flir_label, self.flir_frame)

            # Update XIMEA camera display
            if self.ximea_frame is not None:
                self.set_pixmap(self.ximea_label, self.ximea_frame)

        # 更新FPS显示
        self.fps_label.setText(f"FPS: {self.fps:.1f}")

    def save_snapshot(self):
        """Save snapshots from all cameras"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with self.lock:
            if self.daheng_frame is not None:
                cv2.imwrite(f"daheng_{timestamp}.jpg", self.daheng_frame)
            if self.annotated_frame is not None:
                cv2.imwrite(f"yolo_{timestamp}.jpg", self.annotated_frame)
            if self.flir_frame is not None:
                cv2.imwrite(f"flir_{timestamp}.jpg", self.flir_frame)
            if self.ximea_frame is not None:
                cv2.imwrite(f"ximea_{timestamp}.png", self.ximea_frame)
        self.statusBar().showMessage(f"截图已保存 {timestamp}", 3000)

    def closeEvent(self, event):
        """Stop threads and release resources safely"""
        self.exit_flag = True

        # 等待所有线程退出
        for t in [self.daheng_thread, self.infer_thread, self.flir_thread, self.ximea_thread]:
            t.join(timeout=1.0)

        # 安全释放资源
        try:
            self.cam.stream_off()
            self.cam.close_device()
        except Exception:
            pass

        try:
            self.flir_cam.EndAcquisition()
            self.flir_cam.DeInit()
            self.flir_cam = None
            self.flir_system.ReleaseInstance()
        except Exception:
            pass

        try:
            self.ximea_cam.stop_acquisition()
            self.ximea_cam.close_device()
        except Exception:
            pass

        cv2.destroyAllWindows()
        event.accept()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())