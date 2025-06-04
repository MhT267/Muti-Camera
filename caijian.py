import cv2
import os

# ====== 1. 输入参数 ======
video_path = "demo1.mp4"
output_path = "demo_rgb_1600_2200.mp4"
output_fps = 15  # ✅ 设置你想保存的帧率，比如 15 或 10
# ✅ 设置裁剪帧区间（单位：帧）
start_frame = 1600
end_frame = 2200  # 比如前300帧
# ==========================

cap = cv2.VideoCapture(video_path)

# 视频参数
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 初始红框位置
x1, y1, x2, y2 = 550, 100, 1630, 750

cv2.namedWindow("Adjust ROI")

def nothing(x): pass

cv2.createTrackbar("x1", "Adjust ROI", x1, W, nothing)
cv2.createTrackbar("y1", "Adjust ROI", y1, H, nothing)
cv2.createTrackbar("x2", "Adjust ROI", x2, W, nothing)
cv2.createTrackbar("y2", "Adjust ROI", y2, H, nothing)

print("🎮 调整红框，按 s 保存当前坐标并裁剪视频，按 q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    x1 = cv2.getTrackbarPos("x1", "Adjust ROI")
    y1 = cv2.getTrackbarPos("y1", "Adjust ROI")
    x2 = cv2.getTrackbarPos("x2", "Adjust ROI")
    y2 = cv2.getTrackbarPos("y2", "Adjust ROI")

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Adjust ROI", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        print(f"✅ 选定坐标：x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        break
    elif key == ord('q'):
        print("❌ 未保存，退出")
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

cap.release()
cv2.destroyAllWindows()

# ====== 2. 裁剪指定帧区间和区域 ======

print("🎬 开始裁剪视频中红框区域...")

if os.path.exists(output_path):
    print(f"⚠️ 文件已存在：{output_path}，跳过裁剪。")
    exit(0)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

out_w, out_h = x2 - x1, y2 - y1
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (out_w, out_h))

frame_id = start_frame
while frame_id < end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[y1:y2, x1:x2]
    out.write(roi)

    frame_id += 1
    if frame_id % 30 == 0:
        print(f"▶ 处理帧：{frame_id}/{end_frame}")

cap.release()
out.release()
print(f"✅ 裁剪完成：{output_path}（帧区间：{start_frame} ~ {end_frame}）")
