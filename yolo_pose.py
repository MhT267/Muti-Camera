import gxipy as gx
import cv2
import numpy as np
import sys
import time
import threading
from ultralytics import YOLO
from datetime import datetime

# 固定输出分辨率
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

# 加载模型并 warm-up（推荐）
model = YOLO("./checkpoints/yolo11n-pose.pt")
model.predict(np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8), imgsz=416, device='0', half=True)

# 初始化大恒相机
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num == 0:
    print("未检测到大恒相机")
    sys.exit(1)

str_sn = dev_info_list[0].get("sn")
cam = device_manager.open_device_by_sn(str_sn)
cam.stream_on()

# 共享变量
frame_lock = threading.Lock()
current_frame = None
annotated_frame = None
exit_flag = False

# 推理线程函数
def inference_thread():
    global current_frame, annotated_frame, exit_flag
    while not exit_flag:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        # 推理
        results = model.predict(
            frame,
            imgsz=416,
            device='0',
            conf=0.6,
            half=True,
            verbose=False
        )

        result = results[0]
        rendered = result.plot(conf=True, boxes=True, labels=True, kpt_line=True)

        with frame_lock:
            annotated_frame = rendered

# 启动推理线程
thread = threading.Thread(target=inference_thread)
thread.daemon = True
thread.start()

try:
    frame_count = 0
    start_time = time.time()

    while True:
        raw_image = cam.data_stream[0].get_image()
        if raw_image is None:
            continue

        rgb_image = raw_image.convert("RGB")
        numpy_image = rgb_image.get_numpy_array()
        if numpy_image is None:
            continue

        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(numpy_image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        # 每隔2帧送一帧进推理线程
        if frame_count % 2 == 0:
            with frame_lock:
                current_frame = resized_image.copy()

        # 显示处理后的图像
        with frame_lock:
            display_frame = annotated_frame.copy() if annotated_frame is not None else resized_image.copy()

        # 帧率计算
        fps = 1.0 / (time.time() - start_time)
        start_time = time.time()
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Daheng + YOLO Pose (MultiThreaded)", display_frame)
        frame_count += 1

        # 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    exit_flag = True
    thread.join()
    cam.stream_off()
    cam.close_device()
    cv2.destroyAllWindows()
