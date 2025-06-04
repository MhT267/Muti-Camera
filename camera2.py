import sys
import time
import cv2
import numpy as np
import threading
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton,
                             QWidget, QHBoxLayout, QVBoxLayout,
                             QMainWindow, QStatusBar)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import gxipy as gx
import PySpin

# 初始化模型 - 修正了这里的括号问题
model = YOLO("./checkpoints/yolov8x.pt ")

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧厨房")
        self.setGeometry(100, 100, 2560, 720)  # 适合双屏并列显示

        # 视频参数配置
        self.display_width = 1280
        self.display_height = 720
        self.VIDEO_FORMAT = "MJPG"
        self.FPS = 30

        # 摄像头共享数据
        self.daheng_frame = None
        self.flir_frame = None
        self.annotated_frame = None
        self.lock = threading.Lock()
        self.exit_flag = False
        self.fps = 0

        # 视频录制相关
        self.is_recording = False
        self.daheng_writer = None
        self.flir_writer = None
        self.record_start_time = None

        # 初始化界面
        self.init_ui()

        # 初始化相机
        self.init_daheng()
        self.init_flir()

        # 启动线程
        self.daheng_thread = threading.Thread(target=self.update_daheng)
        self.infer_thread = threading.Thread(target=self.inference)
        self.flir_thread = threading.Thread(target=self.update_flir)
        for t in [self.daheng_thread, self.infer_thread, self.flir_thread]:
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

        # 大恒相机区域
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

        # 合并主布局
        main_layout.addLayout(daheng_layout)
        main_layout.addLayout(flir_layout)

        # 控制按钮区域
        self.fps_label = QLabel("FPS: 0.0")
        self.save_button = QPushButton("保存截图")
        self.record_button = QPushButton("开始录制")
        self.record_button.setCheckable(True)
        self.exit_button = QPushButton("退出")

        self.save_button.clicked.connect(self.save_frames)
        self.record_button.clicked.connect(self.toggle_recording)
        self.exit_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.fps_label)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.record_button)
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

        # 样式设置 - 纯字体透明背景版本
        title_style = """
            QLabel {
                font: bold 24pt "Times New Roman", "SimSun";
                color: black;  /* 黑色文字 */
                background-color: transparent;  /* 完全透明背景 */
                padding: 0px;  /* 去除内边距 */
                margin: 0px;   /* 去除外边距 */
                border: none;  /* 无边框 */
            }
        """
        self.daheng_title.setStyleSheet(title_style)
        self.flir_title.setStyleSheet(title_style)

    def init_daheng(self):
        self.manager = gx.DeviceManager()
        dev_num, dev_info_list = self.manager.update_device_list()
        if dev_num == 0:
            print("大恒相机未连接")
            sys.exit(1)

        self.cam = self.manager.open_device_by_sn(dev_info_list[0].get("sn"))
        self.cam.stream_on()

    def init_flir(self):
        self.flir_system = PySpin.System.GetInstance()
        cam_list = self.flir_system.GetCameras()
        if cam_list.GetSize() == 0:
            print("未检测到FLIR相机")
            sys.exit(1)
        self.flir_cam = cam_list[0]
        self.flir_cam.Init()
        self.flir_cam.BeginAcquisition()

    def update_daheng(self):
        while not self.exit_flag:
            raw = self.cam.data_stream[0].get_image()
            if raw is None:
                continue
            rgb = raw.convert("RGB").get_numpy_array()
            resized = cv2.resize(rgb, (self.display_width, self.display_height))
            with self.lock:
                self.daheng_frame = resized.copy()
            cv2.waitKey(1)

    def update_flir(self):
        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        while not self.exit_flag:
            image = self.flir_cam.GetNextImage(1000)
            if image.IsIncomplete():
                continue
            converted = processor.Convert(image, PySpin.PixelFormat_Mono8)
            array = np.frombuffer(converted.GetData(), dtype=np.uint8).reshape(
                converted.GetHeight(), converted.GetWidth())
            image.Release()
            resized = cv2.resize(array, (self.display_width, self.display_height))
            color_mono = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            with self.lock:
                self.flir_frame = color_mono
            cv2.waitKey(1)

    def inference(self):
        last_time = time.time()
        while not self.exit_flag:
            with self.lock:
                if self.daheng_frame is None:
                    continue
                img = self.daheng_frame.copy()

            # 推理
            results = model.predict(img, imgsz=672, device='0',
                                    half=True, conf=0.3, verbose=False, max_det=10)
            self.annotated_frame = results[0].plot(
                conf=True, boxes=True, labels=True, kpt_line=True)
            now = time.time()
            self.fps = 1.0 / (now - last_time)
            last_time = now

    def toggle_recording(self):
        if self.record_button.isChecked():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fourcc = cv2.VideoWriter_fourcc(*'H264')

            self.daheng_writer = cv2.VideoWriter(
                f"daheng_{timestamp}.avi",
                fourcc,
                self.FPS,
                (self.display_width, self.display_height))

            self.flir_writer = cv2.VideoWriter(
                f"flir_{timestamp}.avi",
                fourcc,
                self.FPS,
                (self.display_width, self.display_height))

            self.record_button.setText("停止录制")
            self.is_recording = True
            self.record_start_time = time.time()
            self.statusBar().showMessage("录制已开始")
        else:
            if self.daheng_writer is not None:
                self.daheng_writer.release()
            if self.flir_writer is not None:
                self.flir_writer.release()

            self.record_button.setText("开始录制")
            self.is_recording = False
            duration = time.time() - self.record_start_time
            self.statusBar().showMessage(
                f"录制已完成，时长: {int(duration)}秒")

    def update_gui(self):
        with self.lock:
            # 更新大恒相机显示
            daheng_display = self.annotated_frame if self.annotated_frame is not None else self.daheng_frame
            if daheng_display is not None:
                self.set_pixmap(self.daheng_label, daheng_display)
                if self.is_recording:
                    resized = cv2.resize(daheng_display,
                                         (self.display_width, self.display_height))
                    self.daheng_writer.write(resized)

            # 更新FLIR相机显示
            if self.flir_frame is not None:
                self.set_pixmap(self.flir_label, self.flir_frame)
                if self.is_recording:
                    resized = cv2.resize(self.flir_frame,
                                         (self.display_width, self.display_height))
                    self.flir_writer.write(resized)

        # 更新FPS显示
        self.fps_label.setText(f"FPS: {self.fps:.1f}")

        # 更新录制时间显示
        if self.is_recording:
            duration = time.time() - self.record_start_time
            self.record_button.setText(
                f"停止录制 ({int(duration)}s)")

    def set_pixmap(self, label, frame):
        rgb = frame
        h, w = rgb.shape[:2]

        # 计算保持比例的缩放尺寸
        target_width = label.width()
        target_height = int(h * (target_width / w))

        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            target_width, target_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)


    def save_frames(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        with self.lock:
            if self.daheng_frame is not None:
                cv2.imwrite(f"daheng_{timestamp}.jpg", self.daheng_frame)
            if self.flir_frame is not None:
                cv2.imwrite(f"flir_{timestamp}.jpg", self.flir_frame)
        self.statusBar().showMessage(f"截图已保存 {timestamp}", 3000)

    def closeEvent(self, event):
        self.exit_flag = True
        time.sleep(0.5)

        # 停止录制
        if self.is_recording:
            self.toggle_recording()

        # 释放相机资源
        self.cam.stream_off()
        self.cam.close_device()
        self.flir_cam.EndAcquisition()
        self.flir_cam.DeInit()
        self.flir_cam = None
        self.flir_system.ReleaseInstance()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())