import os
import cv2
from ultralytics import YOLO

# 输出文件路径
output_path = "demo1_v8.mp4"

# ✅ 如果文件已存在，则跳过整个处理
if os.path.exists(output_path):
    print(f"✅ 文件已存在：{output_path}，跳过处理。")
    exit(0)

# ✅ 加载 YOLOv8 模型
model_pose = YOLO("yolov8x-pose.pt")
model_detect = YOLO("yolov8x.pt")

video_path = "demo1.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    print("⚠️ FPS 获取失败，默认设置为 30")
    fps = 30

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

# 局部区域（ROI）设置
x1, y1, x2, y2 = 500, 100, 1670, 750
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"✅ 全部处理完成，共处理 {frame_id} 帧")
        break

    frame_id += 1
    print(f"▶ 正在处理第 {frame_id} 帧")

    roi = frame[y1:y2, x1:x2]
    rendered_roi = roi.copy()

    try:
        results_detect = model_detect(roi, conf=0.25, verbose=False, device='0')
        result_det = results_detect[0]

        for box in result_det.boxes:
            cls_id = int(box.cls)
            if cls_id == 0:
                continue
            conf = float(box.conf)
            xA, yA, xB, yB = map(int, box.xyxy.tolist()[0])
            label = f"{model_detect.names[cls_id]} {conf:.2f}"
            cv2.rectangle(rendered_roi, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.putText(rendered_roi, label, (xA, yA - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        results_pose = model_pose(roi, conf=0.3, verbose=False, device='0')
        rendered_roi = results_pose[0].plot(img=rendered_roi)

        frame[y1:y2, x1:x2] = rendered_roi

    except Exception as e:
        print(f"❌ 第 {frame_id} 帧处理失败：{e}")
        continue

    out.write(frame)

cap.release()
out.release()
print(f"🎉 YOLOv8 视频处理完成，输出保存为：{output_path}")
