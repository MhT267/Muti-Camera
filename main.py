import sys
import os
import time
import threading
import queue
import ctypes

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QWidget,
    QHBoxLayout, QVBoxLayout, QMainWindow, QSizePolicy, QGridLayout,
    QSpinBox, QGroupBox, QFormLayout, QDoubleSpinBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

# Camera SDKs
import gxipy as gx
import PySpin


class TripleCameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RGB-P-T")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.95), int(screen.height() * 0.88))

        self.lock = threading.Lock()
        self.exit_flag = False

        # --- Display size for GUI preview only ---
        self.display_size = (1280, 720)

        # --- Output dirs ---
        self.out_root = "result"

        # --- Raw frames ---
        self.rgb_raw = None
        self.flir_pol_raw_u12 = None
        self.guide_raw_u16 = None
        self.guide_vis16 = None

        # --- Display frames ---
        self.rgb_disp = None
        self.flir_disp8 = None
        self.guide_disp8 = None
        self.dolp_disp8 = None

        # --- DoLP compute state ---
        self._dolp_busy = False

        # --- FPS ---
        self._last_fps_t = time.time()
        self._fps_counter = 0

        # --- Guide queue (latest frame only) ---
        self._guide_queue = queue.Queue(maxsize=1)
        self._guide_queue_lock = threading.Lock()

        # Hold callback refs to avoid GC
        self._guide_on_frame_cb = None
        self._guide_on_connect_cb = None
        self._guide_on_serial_cb = None

        # SDK handles
        self.manager = None
        self.cam_rgb = None

        self.flir_system = None
        self.flir_cam = None

        self.guide_lib = None

        # UI
        self.init_ui()

        # Init all cameras
        self.init_daheng()
        self.init_flir()
        self.init_guide()

        # After init, try read current exposure and WB and reflect in UI
        #self.refresh_exposure_ui_from_cameras()
        #self.refresh_wb_ui_from_camera()
        self.apply_rgb_exposure_from_ui()
        self.apply_pol_exposure_from_ui()
        self.apply_rgb_wb_manual()

        # Start threads
        self.daheng_thread = threading.Thread(target=self.update_daheng, daemon=True)
        self.flir_thread = threading.Thread(target=self.update_flir, daemon=True)
        self.guide_thread = threading.Thread(target=self.update_guide, daemon=True)
        self.daheng_thread.start()
        self.flir_thread.start()
        self.guide_thread.start()

        # GUI timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)

    # ---------------- UI ----------------
    def init_ui(self):
        self.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-size: 14pt;
                padding: 8px 15px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #256628; }
            QPushButton:disabled { background-color: #888; }
            QLabel { font-size: 14pt; }
            QSpinBox, QDoubleSpinBox { font-size: 14pt; padding: 4px; }
            QGroupBox { font-size: 12pt; font-weight: bold; border: 1px solid #aaa; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }
        """)

        central_layout = QVBoxLayout()

        # 2x2 display grid
        grid = QGridLayout()

        self.rgb_label = QLabel("RGB")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.flir_label = QLabel("POL")
        self.flir_label.setAlignment(Qt.AlignCenter)
        self.flir_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.guide_label = QLabel("IR")
        self.guide_label.setAlignment(Qt.AlignCenter)
        self.guide_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.dolp_label = QLabel("DoLP")
        self.dolp_label.setAlignment(Qt.AlignCenter)
        self.dolp_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid.addWidget(self.rgb_label, 0, 0)
        grid.addWidget(self.flir_label, 0, 1)
        grid.addWidget(self.guide_label, 1, 0)
        grid.addWidget(self.dolp_label, 1, 1)

        # --- Exposure controls ---
        exp_layout = QHBoxLayout()

        self.rgb_exp_group = QGroupBox("RGB 曝光 (us)")
        self.pol_exp_group = QGroupBox("POL 曝光 (us)")

        rgb_form = QFormLayout()
        pol_form = QFormLayout()

        self.rgb_exp_spin = QSpinBox()
        self.rgb_exp_spin.setRange(10, 2_000_000)  # 10us ~ 2s
        self.rgb_exp_spin.setSingleStep(100)
        self.rgb_exp_spin.setValue(40000)

        self.pol_exp_spin = QSpinBox()
        self.pol_exp_spin.setRange(10, 2_000_000)
        self.pol_exp_spin.setSingleStep(100)
        self.pol_exp_spin.setValue(15000)

        self.apply_rgb_exp_btn = QPushButton("应用 RGB 曝光")
        self.apply_pol_exp_btn = QPushButton("应用 POL 曝光")

        # Apply on button
        self.apply_rgb_exp_btn.clicked.connect(self.apply_rgb_exposure_from_ui)
        self.apply_pol_exp_btn.clicked.connect(self.apply_pol_exposure_from_ui)
        self.rgb_exp_spin.editingFinished.connect(self.apply_rgb_exposure_from_ui)
        self.pol_exp_spin.editingFinished.connect(self.apply_pol_exposure_from_ui)

        rgb_form.addRow(QLabel("Time:"), self.rgb_exp_spin)
        rgb_form.addRow(self.apply_rgb_exp_btn)
        self.rgb_exp_group.setLayout(rgb_form)

        pol_form.addRow(QLabel("Time:"), self.pol_exp_spin)
        pol_form.addRow(self.apply_pol_exp_btn)
        self.pol_exp_group.setLayout(pol_form)

        exp_layout.addWidget(self.rgb_exp_group)
        exp_layout.addWidget(self.pol_exp_group)

        # --- White Balance Controls --- ### <--- 修改布局开始
        wb_layout = QHBoxLayout()
        self.rgb_wb_group = QGroupBox("RGB 白平衡 (White Balance)")

        # 1. Manual R/G/B SpinBoxes (Layout)
        self.wb_r_spin = QDoubleSpinBox()
        self.wb_r_spin.setRange(0.0, 15.0)
        self.wb_r_spin.setSingleStep(0.1)
        self.wb_r_spin.setValue(1.43)  # 保留你设置的默认值

        self.wb_g_spin = QDoubleSpinBox()
        self.wb_g_spin.setRange(0.0, 15.0)
        self.wb_g_spin.setSingleStep(0.1)
        self.wb_g_spin.setValue(1.0)

        self.wb_b_spin = QDoubleSpinBox()
        self.wb_b_spin.setRange(0.0, 15.0)
        self.wb_b_spin.setSingleStep(0.1)
        self.wb_b_spin.setValue(2.64)  # 保留你设置的默认值

        wb_manual_layout = QHBoxLayout()
        wb_manual_layout.addWidget(QLabel("R:"))
        wb_manual_layout.addWidget(self.wb_r_spin)
        wb_manual_layout.addWidget(QLabel("G:"))
        wb_manual_layout.addWidget(self.wb_g_spin)
        wb_manual_layout.addWidget(QLabel("B:"))
        wb_manual_layout.addWidget(self.wb_b_spin)

        # 2. Buttons (Auto & Apply) - New Horizontal Layout
        wb_btn_layout = QHBoxLayout()

        self.btn_wb_auto = QPushButton("自动白平衡")
        self.btn_wb_auto.setToolTip("请将相机对准白色区域，然后点击此按钮")
        self.btn_wb_auto.clicked.connect(self.trigger_rgb_wb_auto)

        self.btn_wb_apply = QPushButton("应用手动增益")
        self.btn_wb_apply.clicked.connect(self.apply_rgb_wb_manual)

        # 将两个按钮加入水平布局，参数 '1' 表示 stretch 比例为 1:1，即左右对半
        wb_btn_layout.addWidget(self.btn_wb_auto, 1)
        wb_btn_layout.addWidget(self.btn_wb_apply, 1)

        # 3. Combine into Group Vertical Layout
        wb_v_layout = QVBoxLayout()
        wb_v_layout.addLayout(wb_manual_layout)  # 上面是输入框
        wb_v_layout.addLayout(wb_btn_layout)  # 下面是并排的两个按钮

        self.rgb_wb_group.setLayout(wb_v_layout)
        wb_layout.addWidget(self.rgb_wb_group)
        # ----------------------------------------------- ### <--- 修改布局结束

        # Buttons
        button_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0.0")

        self.dolp_button = QPushButton("计算当前DoLP")
        self.save_button = QPushButton("保存当前画面")
        self.exit_button = QPushButton("退出")

        self.dolp_button.clicked.connect(self.compute_dolp_current_frame)
        self.save_button.clicked.connect(self.save_frames)
        self.exit_button.clicked.connect(self.close)

        button_layout.addWidget(self.fps_label)
        button_layout.addStretch(1)
        button_layout.addWidget(self.dolp_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.exit_button)

        central_layout.addLayout(grid)
        central_layout.addLayout(exp_layout)
        central_layout.addLayout(wb_layout)
        central_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(central_layout)
        self.setCentralWidget(container)
        self.statusBar().setStyleSheet("font: 12pt;")

    # ---------------- White Balance Logic (Daheng) ----------------
    def refresh_wb_ui_from_camera(self):
        """Read current RGB gains from Daheng camera and update spinboxes."""
        if self.cam_rgb is None:
            return

        try:
            r, g, b = self._daheng_get_wb_ratios()
            self.wb_r_spin.blockSignals(True)
            self.wb_g_spin.blockSignals(True)
            self.wb_b_spin.blockSignals(True)

            self.wb_r_spin.setValue(r)
            self.wb_g_spin.setValue(g)
            self.wb_b_spin.setValue(b)

            self.wb_r_spin.blockSignals(False)
            self.wb_g_spin.blockSignals(False)
            self.wb_b_spin.blockSignals(False)
        except Exception as e:
            print(f"Warning reading WB: {e}")

    def trigger_rgb_wb_auto(self):
        """Trigger 'Once' auto white balance."""
        if self.cam_rgb is None:
            return

        self.statusBar().showMessage("正在执行自动白平衡...请稍候", 2000)
        try:
            if hasattr(self.cam_rgb, "BalanceWhiteAuto"):
                try:
                    self.cam_rgb.BalanceWhiteAuto.set(2)
                except:
                    try:
                        self.cam_rgb.BalanceWhiteAuto.set("Once")
                    except:
                        self.statusBar().showMessage("❌ 相机不支持自动白平衡指令", 3000)
                        return

                QTimer.singleShot(1000, self._finish_wb_auto)

        except Exception as e:
            self.statusBar().showMessage(f"❌ 自动白平衡失败: {e}", 3000)

    def _finish_wb_auto(self):
        """Callback after waiting for auto WB."""
        self.refresh_wb_ui_from_camera()
        self.statusBar().showMessage("✅ 自动白平衡完成，已更新参数", 3000)

    def apply_rgb_wb_manual(self):
        """Apply the R, G, B values from spinboxes."""
        if self.cam_rgb is None:
            return

        r = self.wb_r_spin.value()
        g = self.wb_g_spin.value()
        b = self.wb_b_spin.value()

        try:
            if hasattr(self.cam_rgb, "BalanceWhiteAuto"):
                try:
                    self.cam_rgb.BalanceWhiteAuto.set(0)
                except:
                    pass

            self._daheng_set_wb_ratios(r, g, b)
            self.statusBar().showMessage(f"✅ 白平衡已设置: R={r:.2f}, G={g:.2f}, B={b:.2f}", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"❌ 设置白平衡失败: {e}", 3000)

    def _daheng_get_wb_ratios(self):
        r, g, b = 1.0, 1.0, 1.0
        if hasattr(self.cam_rgb, "BalanceRatioSelector") and hasattr(self.cam_rgb, "BalanceRatio"):
            self.cam_rgb.BalanceRatioSelector.set(0)
            r = self.cam_rgb.BalanceRatio.get()
            self.cam_rgb.BalanceRatioSelector.set(1)
            g = self.cam_rgb.BalanceRatio.get()
            self.cam_rgb.BalanceRatioSelector.set(2)
            b = self.cam_rgb.BalanceRatio.get()
        return r, g, b

    def _daheng_set_wb_ratios(self, r, g, b):
        if hasattr(self.cam_rgb, "BalanceRatioSelector") and hasattr(self.cam_rgb, "BalanceRatio"):
            self.cam_rgb.BalanceRatioSelector.set(0)
            self.cam_rgb.BalanceRatio.set(float(r))
            self.cam_rgb.BalanceRatioSelector.set(1)
            self.cam_rgb.BalanceRatio.set(float(g))
            self.cam_rgb.BalanceRatioSelector.set(2)
            self.cam_rgb.BalanceRatio.set(float(b))

    # ---------------- Exposure control helpers ----------------
    def refresh_exposure_ui_from_cameras(self):
        try:
            if self.cam_rgb is not None:
                exp = self._daheng_get_exposure_time_us()
                if exp is not None:
                    self.rgb_exp_spin.setValue(int(round(exp)))
        except Exception:
            pass
        try:
            if self.flir_cam is not None:
                exp = self._flir_get_exposure_time_us()
                if exp is not None:
                    self.pol_exp_spin.setValue(int(round(exp)))
        except Exception:
            pass

    def apply_rgb_exposure_from_ui(self):
        us = int(self.rgb_exp_spin.value())
        ok, msg = self._daheng_set_exposure_time_us(us)
        self.statusBar().showMessage(("✅ " if ok else "⚠️ ") + msg, 3500)

    def apply_pol_exposure_from_ui(self):
        us = int(self.pol_exp_spin.value())
        ok, msg = self._flir_set_exposure_time_us(us)
        self.statusBar().showMessage(("✅ " if ok else "⚠️ ") + msg, 3500)

    # ----- Daheng (gxipy) exposure -----
    def _daheng_try_set_auto_off(self):
        cam = self.cam_rgb
        candidates = ["ExposureAuto", "AutoExposure", "ExposureAutoMode"]
        for name in candidates:
            if hasattr(cam, name):
                node = getattr(cam, name)
                try:
                    if hasattr(node, "set"):
                        node.set(0)
                    else:
                        node(0)
                    return True
                except Exception:
                    continue
        return False

    def _daheng_get_exposure_time_us(self):
        cam = self.cam_rgb
        candidates = ["ExposureTime", "ExposureTimeAbs", "ExposureTimeValue"]
        for name in candidates:
            if hasattr(cam, name):
                node = getattr(cam, name)
                try:
                    if hasattr(node, "get"):
                        return float(node.get())
                    if callable(node):
                        return float(node())
                except Exception:
                    continue
        return None

    def _daheng_set_exposure_time_us(self, us: int):
        if self.cam_rgb is None:
            return False, "Daheng 未初始化"
        cam = self.cam_rgb
        self._daheng_try_set_auto_off()

        lo = hi = None
        try:
            if hasattr(cam, "ExposureTime") and hasattr(cam.ExposureTime, "get_range"):
                lo, hi = cam.ExposureTime.get_range()
        except Exception:
            lo = hi = None

        val = float(us)
        if lo is not None and hi is not None:
            val = float(min(max(val, float(lo)), float(hi)))

        candidates = ["ExposureTime", "ExposureTimeAbs", "ExposureTimeValue"]
        for name in candidates:
            if hasattr(cam, name):
                node = getattr(cam, name)
                try:
                    if hasattr(node, "set"):
                        node.set(val)
                        return True, f"RGB ExposureTime = {val:.0f} us"
                    node(val)
                    return True, f"RGB ExposureTime = {val:.0f} us"
                except Exception:
                    continue
        return False, "RGB 曝光设置失败"

    # ----- FLIR (PySpin) exposure -----
    def _flir_get_nodemap(self):
        if self.flir_cam is None:
            return None
        return self.flir_cam.GetNodeMap()

    def _flir_set_auto_off(self, nodemap):
        try:
            exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
            if PySpin.IsAvailable(exp_auto) and PySpin.IsWritable(exp_auto):
                entry_off = exp_auto.GetEntryByName('Off')
                if PySpin.IsAvailable(entry_off) and PySpin.IsReadable(entry_off):
                    exp_auto.SetIntValue(entry_off.GetValue())
                    return True
        except Exception:
            pass
        return False

    def _flir_get_exposure_time_us(self):
        try:
            nodemap = self._flir_get_nodemap()
            if nodemap is None:
                return None
            exp = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
            if PySpin.IsAvailable(exp) and PySpin.IsReadable(exp):
                return float(exp.GetValue())
        except Exception:
            pass
        return None

    def _flir_set_exposure_time_us(self, us: int):
        if self.flir_cam is None:
            return False, "FLIR 未初始化"
        try:
            nodemap = self._flir_get_nodemap()
            if nodemap is None:
                return False, "FLIR nodemap 不可用"

            self._flir_set_auto_off(nodemap)

            exp = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
            if not (PySpin.IsAvailable(exp) and PySpin.IsWritable(exp)):
                return False, "FLIR ExposureTime 节点不可写"

            lo = float(exp.GetMin())
            hi = float(exp.GetMax())
            val = float(min(max(float(us), lo), hi))
            exp.SetValue(val)
            return True, f"POL ExposureTime = {val:.0f} us"
        except Exception as e:
            return False, f"POL 曝光设置失败：{e}"

    # ---------------- DoLP ----------------
    def compute_dolp_current_frame(self):
        if self._dolp_busy:
            self.statusBar().showMessage("DoLP 计算中…请稍等", 2000)
            return

        with self.lock:
            if self.flir_pol_raw_u12 is None:
                self.statusBar().showMessage("还没有收到偏振原始帧，无法计算 DoLP", 3000)
                return
            u12 = self.flir_pol_raw_u12.copy()

        self._dolp_busy = True
        self.dolp_button.setEnabled(False)
        self.statusBar().showMessage("开始计算 DoLP…", 2000)

        def worker(u12_local: np.ndarray):
            try:
                import polanalyser as pa

                img_000, img_045, img_090, img_135 = pa.demosaicing(u12_local, pa.COLOR_PolarMono)
                image_list = [img_000, img_045, img_090, img_135]
                angles = np.deg2rad([0, 45, 90, 135])
                img_stokes = pa.calcStokes(image_list, angles)
                img_dolp = pa.cvtStokesToDoLP(img_stokes)

                dolp8 = np.clip(img_dolp * 255.0, 0, 255).astype(np.uint8)

                dolp_rgb = None
                try:
                    if hasattr(cv2, "COLORMAP_VIRIDIS"):
                        dolp_bgr = cv2.applyColorMap(dolp8, cv2.COLORMAP_VIRIDIS)
                        dolp_rgb = cv2.cvtColor(dolp_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    dolp_rgb = None

                if dolp_rgb is None:
                    dolp_rgb = np.stack([dolp8, dolp8, dolp8], axis=-1)

                dolp_rgb_rs = cv2.resize(dolp_rgb, self.display_size)

                with self.lock:
                    self.dolp_disp8 = dolp_rgb_rs

                self.statusBar().showMessage("✅ DoLP 已更新（基于当前偏振帧）", 3000)
            except Exception as e:
                with self.lock:
                    self.dolp_disp8 = None
                self.statusBar().showMessage(f"DoLP 计算失败：{e}", 5000)
            finally:
                self._dolp_busy = False
                self.dolp_button.setEnabled(True)

        threading.Thread(target=worker, args=(u12,), daemon=True).start()

    # ---------------- Daheng RGB ----------------
    def init_daheng(self):
        self.manager = gx.DeviceManager()
        dev_num, dev_info_list = self.manager.update_device_list()
        if dev_num == 0:
            raise RuntimeError("大恒相机未连接（gxipy未检测到设备）")

        self.cam_rgb = self.manager.open_device_by_sn(dev_info_list[0].get("sn"))
        self.cam_rgb.stream_on()
        self.statusBar().showMessage("✅ Daheng RGB 已启动", 3000)

    def update_daheng(self):
        while not self.exit_flag:
            try:
                raw = self.cam_rgb.data_stream[0].get_image()
                if raw is None:
                    continue
                rgb = raw.convert("RGB").get_numpy_array()
                rgb_disp = cv2.resize(rgb, self.display_size)

                with self.lock:
                    #self.rgb_raw = cv2.resize(rgb, (640,512))
                    self.rgb_raw = rgb
                    self.rgb_disp = rgb_disp
            except Exception:
                time.sleep(0.02)

            cv2.waitKey(1)

    # ---------------- FLIR polarization ----------------
    def init_flir(self):
        self.flir_system = PySpin.System.GetInstance()
        cam_list = self.flir_system.GetCameras()
        if cam_list.GetSize() == 0:
            raise RuntimeError("未检测到 FLIR 相机（PySpin.GetCameras() = 0）")

        self.flir_cam = cam_list[0]
        self.flir_cam.Init()
        nodemap = self.flir_cam.GetNodeMap()

        pix = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        if PySpin.IsAvailable(pix) and PySpin.IsWritable(pix):
            entry = pix.GetEntryByName('Polarized16')
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                pix.SetIntValue(entry.GetValue())

        self.flir_cam.BeginAcquisition()
        self.statusBar().showMessage("✅ FLIR 已启动", 3000)

    def update_flir(self):
        while not self.exit_flag:
            try:
                img = self.flir_cam.GetNextImage(1000)
                if img.IsIncomplete():
                    img.Release()
                    continue

                try:
                    h, w = img.GetHeight(), img.GetWidth()
                    stride = img.GetStride()
                    buf = img.GetData()

                    row_u16 = stride // 2
                    arr16 = np.frombuffer(buf, dtype=np.uint16).reshape(h, row_u16)[:, :w]

                    mx = int(arr16.max())
                    u12 = (arr16 >> 4).astype(np.uint16) if mx > 4095 else arr16

                    # intensity preview: average 2x2 mosaic
                    p00 = u12[0::2, 0::2]
                    p01 = u12[0::2, 1::2]
                    p10 = u12[1::2, 0::2]
                    p11 = u12[1::2, 1::2]
                    I = ((p00.astype(np.uint32) + p01.astype(np.uint32) +
                          p10.astype(np.uint32) + p11.astype(np.uint32)) // 4).astype(np.uint16)

                    disp8 = np.clip(I.astype(np.float32) * (255.0 / 4095.0), 0, 255).astype(np.uint8)
                    disp8_rs = cv2.resize(disp8, self.display_size)

                    with self.lock:
                        self.flir_pol_raw_u12 = u12
                        self.flir_disp8 = disp8_rs
                finally:
                    img.Release()
            except Exception:
                time.sleep(0.02)

            cv2.waitKey(1)

    # ---------------- Guide thermal (DLL callback) ----------------
    def init_guide(self):
        dll_dir = os.path.abspath(r"./checkpoints")
        dll_path = os.path.join(dll_dir, "GuideUSB2LiveStream.dll")
        if not os.path.exists(dll_path):
            raise RuntimeError(f"Guide DLL not found: {dll_path}")

        try:
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
            self.guide_lib = ctypes.WinDLL(dll_path)
        except Exception as e:
            raise RuntimeError(f"加载 Guide DLL 失败: {e}")

        class DeviceStatus(ctypes.c_int):
            DEVICE_CONNECT_OK = 1
            DEVICE_DISCONNECT_OK = -1

        class GuideUsbFrameData(ctypes.Structure):
            _fields_ = [
                ("frame_width", ctypes.c_int),
                ("frame_height", ctypes.c_int),
                ("frame_rgb_data", ctypes.POINTER(ctypes.c_ubyte)),
                ("frame_rgb_data_length", ctypes.c_int),
                ("frame_src_data", ctypes.POINTER(ctypes.c_short)),
                ("frame_src_data_length", ctypes.c_int),
                ("frame_yuv_data", ctypes.POINTER(ctypes.c_short)),
                ("frame_yuv_data_length", ctypes.c_int),
                ("paramLine", ctypes.POINTER(ctypes.c_short)),
                ("paramLine_length", ctypes.c_int),
            ]

        class GuideUsbSerialData(ctypes.Structure):
            _fields_ = [
                ("serial_recv_data", ctypes.POINTER(ctypes.c_ubyte)),
                ("serial_recv_data_length", ctypes.c_int),
            ]

        OnFrameDataCBType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(GuideUsbFrameData))
        OnConnectCBType = ctypes.WINFUNCTYPE(ctypes.c_int, DeviceStatus)
        OnSerialCBType = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(GuideUsbSerialData))

        @OnFrameDataCBType
        def on_frame_data(pVideoData):
            try:
                width = pVideoData.contents.frame_width
                height = pVideoData.contents.frame_height
                length_y16 = pVideoData.contents.frame_src_data_length
                ptr_y16 = pVideoData.contents.frame_src_data

                if ptr_y16 and length_y16 == width * height:
                    y16 = np.ctypeslib.as_array(ptr_y16, shape=(length_y16,))
                    y16 = y16.reshape((height, width)).copy()

                    y16_i32 = y16.astype(np.int32)
                    if y16_i32.min() < 0:
                        y16_i32 = y16_i32 + 65536
                    y16_u16 = np.clip(y16_i32, 0, 65535).astype(np.uint16)

                    with self._guide_queue_lock:
                        if self._guide_queue.full():
                            try:
                                self._guide_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self._guide_queue.put_nowait(y16_u16)
                        except queue.Full:
                            pass
            except Exception:
                pass
            return 0

        @OnConnectCBType
        def on_connect_status(status):
            return 0

        @OnSerialCBType
        def on_serial_data(pSerialData):
            return 0

        self._guide_on_frame_cb = on_frame_data
        self._guide_on_connect_cb = on_connect_status
        self._guide_on_serial_cb = on_serial_data

        lib = self.guide_lib
        lib.guide_usb_initial.restype = ctypes.c_int
        lib.guide_usb_exit.restype = ctypes.c_int
        lib.guide_usb_closestream.restype = ctypes.c_int
        lib.guide_usb_openstream_auto.restype = ctypes.c_int
        lib.guide_usb_openstream_auto.argtypes = [OnFrameDataCBType, OnConnectCBType]
        lib.guide_usb_sendcommand.restype = ctypes.c_int
        lib.guide_usb_sendcommand.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
        lib.guide_usb_getserialdata.restype = ctypes.c_int
        lib.guide_usb_getserialdata.argtypes = [ctypes.c_int, OnSerialCBType]

        ret = lib.guide_usb_initial()
        if ret not in (0, 1):
            raise RuntimeError(f"guide_usb_initial() failed: {ret}")

        lib.guide_usb_getserialdata(1, self._guide_on_serial_cb)

        cmd_low = (ctypes.c_ubyte * 12)(0x55, 0xAA, 0x07, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x0A, 0xF0)
        lib.guide_usb_sendcommand(cmd_low, 12)

        ret = lib.guide_usb_openstream_auto(self._guide_on_frame_cb, self._guide_on_connect_cb)
        if ret != 0:
            raise RuntimeError(f"guide_usb_openstream_auto() failed: {ret}")

        self.statusBar().showMessage("✅ Guide IR 已启动", 3000)

    def update_guide(self):
        last_valid_u16 = None
        while not self.exit_flag:
            try:
                has_new = False
                with self._guide_queue_lock:
                    if not self._guide_queue.empty():
                        last_valid_u16 = self._guide_queue.get_nowait()
                        has_new = True

                if has_new and last_valid_u16 is not None:
                    mn = int(last_valid_u16.min())
                    mx = int(last_valid_u16.max())
                    # print(f"min: {mn}, max: {mx}")
                    if mx > mn:
                        vis = ((last_valid_u16.astype(np.float32) - mn) / (mx - mn) * 255.0).astype(np.uint8)
                        vis16 = ((last_valid_u16.astype(np.float32) - mn) / (mx - mn) * 65535.0).astype(np.uint16)
                    else:
                        vis = np.zeros_like(last_valid_u16, dtype=np.uint8)
                        vis16 = np.zeros_like(last_valid_u16, dtype=np.uint16)

                    vis_rs = cv2.resize(vis, self.display_size)

                    with self.lock:
                        self.guide_raw_u16 = last_valid_u16
                        self.guide_disp8 = vis_rs
                        self.guide_vis16 = vis16

                time.sleep(0.005)
            except Exception:
                time.sleep(0.02)

    # ---------------- GUI update ----------------
    def update_gui(self):
        with self.lock:
            if self.rgb_disp is not None:
                self.set_pixmap(self.rgb_label, self.rgb_disp)
            if self.flir_disp8 is not None:
                self.set_pixmap(self.flir_label, self.flir_disp8)
            if self.guide_disp8 is not None:
                self.set_pixmap(self.guide_label, self.guide_disp8)
            if self.dolp_disp8 is not None:
                self.set_pixmap(self.dolp_label, self.dolp_disp8)

        self._fps_counter += 1
        now = time.time()
        dt = now - self._last_fps_t
        if dt >= 1.0:
            fps = self._fps_counter / dt
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._fps_counter = 0
            self._last_fps_t = now

    def set_pixmap(self, label: QLabel, frame: np.ndarray):
        if frame.ndim == 2:
            h, w = frame.shape
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w = frame.shape[:2]
            qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            max(1, label.width()), max(1, label.height()),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    # ---------------- Save ----------------
    def save_frames(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.out_root, ts)
        os.makedirs(session_dir, exist_ok=True)

        rgb_path = os.path.join(session_dir, "rgb.tiff")
        flir_path = os.path.join(session_dir, "pol.tiff")
        guide_raw_path = os.path.join(session_dir, "raw.npy")
        guide_vis_path = os.path.join(session_dir, "vis.tiff")

        rgb_ok = flir_ok = guide_ok = False

        with self.lock:
            if self.rgb_raw is not None:
                bgr = cv2.cvtColor(self.rgb_raw, cv2.COLOR_RGB2BGR)
                rgb_ok = cv2.imwrite(rgb_path, bgr)

            if self.flir_pol_raw_u12 is not None:
                pol_vis16 = (self.flir_pol_raw_u12.astype(np.uint16) << 4)
                flir_ok = cv2.imwrite(flir_path, pol_vis16)

            if self.guide_raw_u16 is not None and self.guide_vis16 is not None:
                np.save(guide_raw_path, self.guide_raw_u16)
                guide_ok = cv2.imwrite(guide_vis_path, self.guide_vis16)

        msg = (
            f"已保存到: {session_dir} | "
            f"RGB: {'OK' if rgb_ok else 'FAIL'} | "
            f"POL: {'OK' if flir_ok else 'FAIL'} | "
            f"Guide: {'OK' if guide_ok else 'FAIL'}"
        )
        self.statusBar().showMessage(msg, 5000)

    # ---------------- Cleanup ----------------
    def closeEvent(self, event):
        self.exit_flag = True
        time.sleep(0.2)

        try:
            if self.cam_rgb is not None:
                self.cam_rgb.stream_off()
                self.cam_rgb.close_device()
        except Exception:
            pass

        try:
            if self.flir_cam is not None:
                try:
                    self.flir_cam.EndAcquisition()
                except Exception:
                    pass
                self.flir_cam.DeInit()
                self.flir_cam = None
            if self.flir_system is not None:
                self.flir_system.ReleaseInstance()
                self.flir_system = None
        except Exception:
            pass

        try:
            if self.guide_lib is not None:
                try:
                    self.guide_lib.guide_usb_closestream()
                except Exception:
                    pass
                try:
                    self.guide_lib.guide_usb_exit()
                except Exception:
                    pass
                self.guide_lib = None
        except Exception:
            pass

        cv2.destroyAllWindows()
        event.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    try:
        win = TripleCameraApp()
    except Exception as e:
        print("❌ 初始化失败：", e)
        return 1

    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())