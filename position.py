import cv2

video_path = "demo1.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频信息
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始红框位置（可以改成上次的位置）
x1, y1, x2, y2 = 460, 144, 1532, 980

# 创建窗口和滑动条
cv2.namedWindow("Adjust ROI")

def nothing(x):
    pass

# 创建 TrackBars
cv2.createTrackbar("x1", "Adjust ROI", x1, W, nothing)
cv2.createTrackbar("y1", "Adjust ROI", y1, H, nothing)
cv2.createTrackbar("x2", "Adjust ROI", x2, W, nothing)
cv2.createTrackbar("y2", "Adjust ROI", y2, H, nothing)

print("🎮 调整红框，按 s 保存当前坐标，按 q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 获取当前滑动条位置
    x1 = cv2.getTrackbarPos("x1", "Adjust ROI")
    y1 = cv2.getTrackbarPos("y1", "Adjust ROI")
    x2 = cv2.getTrackbarPos("x2", "Adjust ROI")
    y2 = cv2.getTrackbarPos("y2", "Adjust ROI")

    # 确保坐标合法
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # 绘制红框
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 显示帧
    cv2.imshow("Adjust ROI", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        print(f"✅ 选定坐标：x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        break
    elif key == ord('q'):
        print("❌ 未保存，退出")
        break

cap.release()
cv2.destroyAllWindows()
