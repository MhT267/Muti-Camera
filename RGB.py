import gxipy as gx
import cv2
import numpy as np
import sys
import threading
from datetime import datetime

# 固定输出分辨率
OUTPUT_WIDTH = 2448
OUTPUT_HEIGHT = 2048

# 初始化视频写入器
def init_video_writer():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{now}.mp4"
    # 使用H.264编码器（avc1）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(
        filename,
        fourcc,
        30.0,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )

# 视频写入线程
def video_writer_thread(video_writer, frame, recording):
    if video_writer is not None and recording:
        resized_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        video_writer.write(resized_frame)

# 初始化设备管理器
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num == 0:
    print("No devices found")
    sys.exit(1)

# 获取设备并开始采集
str_sn = dev_info_list[0].get("sn")
cam = device_manager.open_device_by_sn(str_sn)
cam.stream_on()

# 视频写入器初始化标志
video_writer = None
recording = False
frame_count = 0
start_time = datetime.now()

try:
    while True:
        raw_image = cam.data_stream[0].get_image()
        if raw_image is None:
            continue

        rgb_image = raw_image.convert("RGB")
        if rgb_image is None:
            continue

        numpy_image = rgb_image.get_numpy_array()
        if numpy_image is None:
            continue

        bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # 显示录制状态
        if recording:
            cv2.putText(bgr_image, "REC", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 实时帧率显示
        frame_count += 1
        elapsed_time = (datetime.now() - start_time).total_seconds()
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(bgr_image, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("Real-Time Image", bgr_image)

        cv2.resizeWindow("Real-Time Image", 2448, 2048)  # 你可以调整为你想要的宽度和高度
        key = cv2.waitKey(1) & 0xFF

        # ESC键：退出
        if key == 27:
            print("ESC pressed, saving and exiting...")
            if video_writer is not None:
                video_writer.release()
            break

        # 空格键：保存当前帧
        elif key == 32:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"frame_{timestamp}.png"
            cv2.imwrite(filename, bgr_image)
            print(f"Saved current frame as {filename}")

        # 自动开始录制
        if video_writer is None:
            video_writer = init_video_writer()
            recording = True

        # 写入压缩后的视频帧（使用线程）
        if video_writer is not None:
            threading.Thread(target=video_writer_thread, args=(video_writer, bgr_image, recording)).start()

finally:
    # 释放资源
    cam.stream_off()
    cam.close_device()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
