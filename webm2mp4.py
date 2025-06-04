import subprocess
import os

def convert_webm_to_mp4(input_path, output_path=None):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在：{input_path}")
        return

    # 自动生成输出路径（如果未指定）
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_converted.mp4"

    # 构造 FFmpeg 命令
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        output_path
    ]

    try:
        print("🚀 正在转换中，请稍等...")
        subprocess.run(command, check=True)
        print(f"✅ 转换完成，输出文件：{output_path}")
    except FileNotFoundError:
        print("❌ 未找到 ffmpeg，请确保它已安装并加入环境变量（Path）")
    except subprocess.CalledProcessError:
        print("❌ 转换失败，请检查 FFmpeg 是否正常运行")

# 示例用法
if __name__ == "__main__":
    input_video = "./video/demo1.webm"         # 替换为你自己的输入文件
    output_video = "./results/demo1.mp4"           # 可选，若不填会自动生成
    convert_webm_to_mp4(input_video, output_video)
