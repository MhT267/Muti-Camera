from ximea import xiapi
import numpy as np
import cv2
import time
from datetime import datetime
import threading
import os

# === 设置基础参数 ===
WIDTH, HEIGHT, SCALE = 2048, 2048, 1
FPS_TARGET = 25

# 两台相机的 SN
SN_LIST = ['04283550', '42289050']

# 相机配置
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

# === 通用工具函数 ===
def extract_raw_channels_no_interp(mosaic_img):
    channels = []
    for dy in range(3):
        for dx in range(3):
            ch = mosaic_img[dy::3, dx::3]
            channels.append(ch)
    return channels

def draw_mosaic_grid_no_interp(channels, labels):
    grid_size = 682
    grid = np.zeros((2048, 2048), dtype=np.uint8)
    for idx, ch in enumerate(channels):
        row, col = divmod(idx, 3)
        y, x = row * grid_size, col * grid_size
        ch_resized = cv2.resize(ch, (grid_size, grid_size))
        grid[y:y + grid_size, x:x + grid_size] = ch_resized
        cv2.putText(grid, f"{labels[idx]}nm", (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    return grid

# === CIE 简化 RGB 权重表 ===
CIE_RGB_TABLE = {
    453: (0.1, 0.2, 0.9), 468: (0.1, 0.3, 0.9), 484: (0.0, 0.5, 0.8),
    496: (0.0, 0.6, 0.7), 513: (0.0, 0.8, 0.4), 530: (0.1, 0.9, 0.2),
    540: (0.3, 0.85, 0.1), 556: (0.5, 0.7, 0.1), 570: (0.7, 0.6, 0.0),
    621: (0.8, 0.4, 0.0), 638: (0.85, 0.3, 0.0), 658: (0.9, 0.2, 0.0),
    677: (0.95, 0.1, 0.0), 698: (0.98, 0.05, 0.0), 719: (1.0, 0.0, 0.0),
    738: (0.9, 0.05, 0.1), 757: (0.8, 0.1, 0.15), 775: (0.7, 0.15, 0.2)
}

def fuse_channels_cie(channels, wavelengths):
    h, w = channels[0].shape
    rgb_f = np.zeros((h, w, 3), dtype=np.float32)
    for ch, wl in zip(channels, wavelengths):
        r, g, b = CIE_RGB_TABLE.get(wl, (0.0, 0.0, 0.0))
        ch_norm = cv2.normalize(ch, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        rgb_f[..., 0] += b * ch_norm
        rgb_f[..., 1] += g * ch_norm
        rgb_f[..., 2] += r * ch_norm
    return np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8)

# === 修改 CameraHandler 类 ===
class CameraHandler:
    def __init__(self, sn):
        self.cam = xiapi.Camera()
        self.cam.open_device_by_SN(sn)
        self.cam.set_exposure(30000)
        self.cam.set_imgdataformat('XI_MONO8')
        self.cam.set_width(WIDTH)
        self.cam.set_height(HEIGHT)
        self.cam.start_acquisition()
        self.img = xiapi.Image()
        self.sn = sn
        self.config = camera_configs[sn]
        self.frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)  # 改为彩色图
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop)
        self.thread.start()

    def capture_loop(self):
        while self.running:
            self.cam.get_image(self.img)
            raw = self.img.get_image_data_numpy()

            # 裁剪为可被3整除
            h, w = raw.shape
            raw = raw[:h - h % 3, :w - w % 3]

            chs_raw = extract_raw_channels_no_interp(raw)
            chs_sorted = [chs_raw[i] for i in self.config["index_map"]]

            # === 新增：生成伪彩RGB图 ===
            cie_rgb = fuse_channels_cie(chs_sorted, self.config["wavelengths"])
            cie_resized = cv2.resize(cie_rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

            with self.lock:
                self.frame = cie_resized.copy()

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cam.stop_acquisition()
        self.cam.close_device()

# === 初始化相机对象 ===
handlers = [CameraHandler(sn) for sn in SN_LIST]
print("双相机伪彩采集已启动，按 Q 退出...")

# === 主显示循环 ===
frame_count, start_time = 0, time.time()
while True:
    frames = [h.get_frame() for h in handlers]
    combined = np.hstack(frames)
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        start_time, frame_count = time.time(), 0
        cv2.putText(combined, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Dual Spectral Pseudo-Color", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 清理资源 ===
for h in handlers:
    h.stop()
cv2.destroyAllWindows()
