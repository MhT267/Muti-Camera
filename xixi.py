import cv2
import numpy as np
from ximea import xiapi
import time
import os
from datetime import datetime

# === 参数配置 ===
WIDTH, HEIGHT = 2048, 2048
SCALE = 1
FPS_TARGET = 25
sn = '42289050'

camera_configs = {
    "42289050": {
        "index_map": [4, 3, 5, 7, 6, 8, 1, 0, 2],
        "wavelengths": [621, 638, 658, 677, 698, 719, 738, 757, 775]
    }
}

# === CIE 简化 RGB 权重表 ===
CIE_RGB_TABLE = {
    621: (0.2, 0.01, 0.0),
    638: (0.35, 0.0, 0.0),
    658: (0.6, 0.0, 0.0),
    677: (0.75, 0.0, 0.0),
    698: (0.85, 0.0, 0.0),
    719: (0.9, 0.0, 0.0),
    738: (0.95, 0.0, 0.0),
    757: (0.9, 0.1, 0.0),
    775: (0.8, 0.15, 0.0)
}

# === 通道提取函数 ===
def extract_raw_channels(mosaic_img):
    """从原始 mosaic 图中提取 9 个通道灰度图，统一裁剪尺寸，避免尺寸不一致"""
    h, w = mosaic_img.shape
    h = h - h % 3  # 向下取整
    w = w - w % 3
    mosaic_img = mosaic_img[:h, :w]  # 裁剪为 3 的整数倍
    channels = []
    for dy in range(3):
        for dx in range(3):
            ch = mosaic_img[dy::3, dx::3]
            channels.append(ch)
    return channels

# === CIE融合函数 ===
def fuse_channels_cie(channels, wavelengths):
    h, w = channels[0].shape
    rgb_f = np.zeros((h, w, 3), dtype=np.float32)
    for ch, wl in zip(channels, wavelengths):
        r, g, b = CIE_RGB_TABLE.get(wl, (0.0, 0.0, 0.0))
        ch_norm = cv2.normalize(ch, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        rgb_f[..., 0] += b * ch_norm
        rgb_f[..., 1] += g * ch_norm
        rgb_f[..., 2] += r * ch_norm
    rgb_img = np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8)
    return rgb_img

# === 相机初始化 ===
cam = xiapi.Camera()
cam.open_device_by_SN(sn)
cam.set_exposure(30000)
cam.set_imgdataformat('XI_MONO8')
cam.set_width(WIDTH)
cam.set_height(HEIGHT)
cam.start_acquisition()
img = xiapi.Image()

# === 加载配置 ===
config = camera_configs[sn]
index_map = config["index_map"]
wavelengths = config["wavelengths"]

# === 输出设置 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("output_9ch", f"record_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
video_path = os.path.join(save_dir, "spectral_cie_fusion.avi")
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), FPS_TARGET, (WIDTH, HEIGHT), isColor=True)

# === 主循环 ===
frame_count = 0
start_time = time.time()
print("采集中，按 Q 退出...")

while True:
    cam.get_image(img)
    raw = img.get_image_data_numpy()
    chs = extract_raw_channels(raw)
    chs_sorted = [chs[i] for i in index_map]
    cie_img = fuse_channels_cie(chs_sorted, wavelengths)
    cie_img = cv2.resize(cie_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    # 帧率显示
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
        cv2.putText(cie_img, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("CIE Spectral Fusion", cie_img)
    writer.write(cie_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 清理 ===
cam.stop_acquisition()
cam.close_device()
writer.release()
cv2.destroyAllWindows()
print(f"保存至: {video_path}")
