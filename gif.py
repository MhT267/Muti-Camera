import cv2
import imageio

def mp4_to_gif(video_path, gif_path, start_time, end_time, resize_width=None, resize_height=None, fps=12):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * original_fps)
    end_frame = int(end_time * original_fps)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_width and resize_height:
            frame_rgb = cv2.resize(frame_rgb, (resize_width, resize_height))
        frames.append(frame_rgb)

    cap.release()

    # 保存为GIF，设置更合理的压缩参数
    imageio.mimsave(
        gif_path,
        frames,
        fps=fps,
        palettesize=64,
        loop=0  # 无限循环
    )

    print(f"✅ GIF已保存，帧数: {len(frames)}, 路径: {gif_path}")


# 示例用法
if __name__ == "__main__":
    mp4_to_gif(
        video_path="sk.mp4",     # 输入视频路径
        gif_path="sk.gif",              # 输出GIF文件名
        start_time=0,                          # 开始秒数
        end_time=20,                           # 结束秒数
        resize_width=960,                      # 可选：宽度
        resize_height=540,                     # 可选：高度
        fps=15                               # 输出GIF帧率
    )
