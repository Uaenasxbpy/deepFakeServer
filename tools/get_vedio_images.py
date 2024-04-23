import cv2
import os
import time

def extract_frames(video_file, output_folder='temp', num_frames=10):
    """
    从视频文件中提取指定数量的帧，并保存为JPEG图片。
    参数:
    video_file: 视频文件路径。
    output_folder: 保存提取帧的文件夹路径，默认为'uploads'。
    num_frames: 要提取的帧的数量，默认为10。
    """
    start_time = time.time()
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("错误：无法打开视频文件。")
        return

    # 获取视频属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps  # 视频总时长（秒）

    # 计算提取帧的间隔
    interval = max(1, total_frames // num_frames)  # 确保间隔至少为1

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 提取帧
    frame_count = 0
    extracted_frames = 0
    while cap.isOpened() and extracted_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_file = os.path.join(output_folder, f"frame_{extracted_frames}.jpg")
            cv2.imwrite(frame_file, frame)
            extracted_frames += 1

        frame_count += 1

        # 释放视频捕获对象
    cap.release()
    response_time = format(time.time() - start_time, '.2f')
    # 返回视频切分所需的时间
    return response_time
