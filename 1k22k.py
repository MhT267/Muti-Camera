import cv2
import os
from realesrgan import RealESRGAN
from PIL import Image
import torch

# 路径设置
input_video = "demo.mp4"
output_video = "demo_realesrgan_2k.mp4"
target_resolution = (2560, 1440)

# 初始化视频
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 初始化输出视频（2K）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, target_resolution)

# 初始化 Real-ESRGAN 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('RealESRGAN_x4plus.pth')  # 会自动下载，无需手动

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV BGR → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 超分辨率增强
    sr_image = model.predict(pil_img)

    # 缩放到指定输出大小（比如2K）
    sr_image = sr_image.resize(target_resolution, Image.BICUBIC)

    # 转回 OpenCV 写入
    result_frame = cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)
    out.write(result_frame)

    frame_id += 1
    if frame_id % 10 == 0:
        print(f"🖼 已处理帧：{frame_id}/{total}")

cap.release()
out.release()
print(f"✅ 使用 Real-ESRGAN 完成超分放大，输出视频：{output_video}")
