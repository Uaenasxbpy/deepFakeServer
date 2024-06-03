import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from concurrent.futures import ThreadPoolExecutor, Future
import shutil
from Model.combine import DeepFakeClassifier
from Model import use_resnet50
from tools import get_vedio_images
from typing import Tuple

# 设置日志记录配置
logging.basicConfig(level=logging.INFO)

# 处理跨域请求
app = Flask(__name__)
CORS(app)

# 设置上传文件夹的路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 临时文件夹，用于存储待处理的文件
TEMP_FOLDER = 'temp'
app.config['TEMP_FOLDER'] = TEMP_FOLDER


def ensure_dir(file_path: str) -> None:
    """
    确保目录存在，如果不存在则创建
    :param file_path: 目录路径
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        logging.info(f'目录 {file_path} 创建成功')
    else:
        logging.info(f'目录 {file_path} 已经存在')


def cleanup_folder(folder_path: str) -> None:
    """
    清理文件夹中的所有文件和文件夹
    :param folder_path: 文件夹路径
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            logging.info(f'删除文件或文件夹 {file_path}')
        except Exception as e:
            logging.error(f'删除文件或文件夹失败 {file_path}. 原因是: {e}')


# 确保两个文件夹都存在
ensure_dir(app.config['UPLOAD_FOLDER'])
ensure_dir(app.config['TEMP_FOLDER'])

executor = ThreadPoolExecutor(max_workers=10)  # 设置线程池大小


def allowed_file(filename: str) -> bool:
    """
    检查文件名称是否可用
    :param filename: 文件名称
    :return: 是否为允许的文件类型
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}
    is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    logging.info(f'文件 {filename} 检查结果: {is_allowed}')
    return is_allowed


def process_image(image_path: str) -> Tuple[bool, str, str]:
    """
    判断图片是否为伪造的
    :param image_path: 图片路径
    :return: 预测结果和准确率信息
    """
    logging.info(f'开始处理图片：{image_path}')
    # logging.info(f"使用的结合模型处理！")
    # classify = DeepFakeClassifier()
    # predicted_class, probability = classify.predict_combined(image_path=image_path)
    logging.info(f'使用单个模型')
    predicted_class, probability = use_resnet50.get_result(image_path=image_path)
    probability_str = f"{probability * 100:.2f}"
    if predicted_class == 'Real':
        result = (True, "照片是真的", f"准确率为: {probability_str}%")
    else:
        result = (True, "照片是假的", f"准确率为: {probability_str}%")
    logging.info(f'处理结果：{result}')
    return result


def process_video(filepath: str) -> Tuple[bool, str, str]:
    """
    判断视频是否为伪造的
    :param filepath: 视频路径
    :return: 预测结果和准确率信息
    """
    logging.info(f'开始处理视频：{filepath}')
    time1, image_paths = get_vedio_images.extract_frames(filepath)
    logging.info(f'提取的视频帧路径：{image_paths}')

    predicted_dict = []
    confidence_dict = []

    for image_path in image_paths:
        # logging.info(f"使用的结合模型处理！")
        # classify = DeepFakeClassifier()
        # predicted_class, probability = classify.predict_combined(image_path=image_path)
        logging.info(f'使用单个模型')
        predicted_class, probability = use_resnet50.get_result(image_path=image_path)
        if predicted_class == 'Real':
            predicted = 1
        else:
            predicted = 0
        confidence = probability
        predicted_dict.append(predicted)
        confidence_dict.append(confidence)

    logging.info(f'预测结果字典：{predicted_dict}')
    logging.info(f'置信度字典：{confidence_dict}')

    count_0 = predicted_dict.count(0)
    count_1 = predicted_dict.count(1)

    indices_0 = [i for i, x in enumerate(predicted_dict) if x == 0]
    indices_1 = [i for i, x in enumerate(predicted_dict) if x == 1]

    avg_confidence_0 = sum(float(confidence_dict[i]) for i in indices_0) / len(indices_0) if indices_0 else 0
    avg_confidence_1 = sum(float(confidence_dict[i]) for i in indices_1) / len(indices_1) if indices_1 else 0

    avg_confidence_0_str = f"{(avg_confidence_0 * 100):.2f}"
    avg_confidence_1_str = f"{(avg_confidence_1 * 100):.2f}"

    logging.info(f'假视频平均置信度：{avg_confidence_0_str}')
    logging.info(f'真视频平均置信度：{avg_confidence_1_str}')

    if count_0 > count_1:
        result = (True, "视频是假的", f"准确率高达: {avg_confidence_0_str}%, 一共使用了 {time1} s。")
    elif count_0 < count_1:
        result = (True, "视频是真的", f"准确率高达: {avg_confidence_1_str}%, 一共使用了 {time1} s。")
    else:
        result = (False, "不能判别", "换一个视频试一试")

    logging.info(f'处理结果：{result}')
    return result


@app.route('/process', methods=['POST'])
def process_file() -> jsonify:
    """
    处理上传的文件，判断其是否为伪造的
    :return: JSON格式的处理结果
    """
    if 'file' not in request.files:
        logging.warning('请求中没有文件部分')
        return jsonify({'status': 0, 'message': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        logging.warning('没有传入文件名')
        return jsonify({'status': 0, 'message': '没有文件传入！'})

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        logging.info(f'文件保存到：{filepath}')

        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # 图片
            future = executor.submit(process_image, filepath)
        elif file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
            # 视频
            future = executor.submit(process_video, filepath)
        else:
            logging.warning(f'不支持的文件类型：{file.filename}')
            return jsonify({'status': 0, 'message': '这个文件类型不支持！'})

        def get_result(future: Future) -> jsonify:
            success, message, precision = future.result()
            logging.info(f'返回结果：success={success}, message={message}, precision={precision}')
            return jsonify({'status': 1, 'message': message, 'precision': precision})

        return get_result(future)
    else:
        logging.warning(f'文件类型不支持：{file.filename}')
        return jsonify({'status': 0, 'message': '文件类型不支持，检查后再传入！'})


@app.teardown_appcontext
def teardown(exc=None) -> None:
    """
    在应用关闭时调用清理函数，确保所有临时文件都被删除
    :param exc: 异常信息
    """
    logging.info('清理临时文件夹')
    cleanup_folder(app.config['TEMP_FOLDER'])
    cleanup_folder(app.config['UPLOAD_FOLDER'])


if __name__ == '__main__':
    logging.info('API 服务器运行在 http://127.0.0.1:5000/process')
    app.run(debug=True, port=5000)
