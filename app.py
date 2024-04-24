import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from concurrent.futures import ThreadPoolExecutor
import shutil

from Model import test
from tools import get_vedio_images
# 处理跨域亲请求
app = Flask(__name__)
CORS(app)

# 设置上传文件夹的路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 临时文件夹，用于存储待处理的文件
TEMP_FOLDER = 'temp'
app.config['TEMP_FOLDER'] = TEMP_FOLDER


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def cleanup_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f'删除文件或文件夹失败 {file_path}. 原因是: {e}')

# 确保两个文件夹都存在
ensure_dir(app.config['UPLOAD_FOLDER'])
ensure_dir(app.config['TEMP_FOLDER'])

executor = ThreadPoolExecutor(max_workers=10)  # 设置线程池大小


def allowed_file(filename):
    '''
    检查文件名称是否可用
    :param filename:
    :return:
    '''
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# TODO 处理图片
def process_image(filepath):
    '''
    判断图片是否为伪造的
    :param filepath:
    :return:
    '''
    logging.info(f'图片路径：{filepath}')
    predicted_label, confidence = test.detect_image(image=filepath)
    return predicted_label, "照片是真的 ", "准确率为:" + str(confidence * 100) + "%"

# TODO 处理视频
def process_video(filepath):
    '''
    判断视频是否为伪造的
    :param filepath:
    :return:
    '''
    time1, image_path = get_vedio_images.extract_frames(filepath)
    print(image_path)
    for frame in image_path:
        process_image(frame)
    return False, "视频是假的", "准确率高达95.35%,一共使用了" + str(time1) + "s。"

# TODO API接口的路由
@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'status': 0, 'message': 'No file part'})
    file = request.files['file']
    print(file)
    # 构建完整的文件路径
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    if file.filename == '':
        return jsonify({'status': 0, 'message': '没有文件传入！'})

    if file and allowed_file(file.filename):
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # 图片
            future = executor.submit(process_image, filepath)
            print(future.result())
        elif file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
            # 视频
            future = executor.submit(process_video, filepath)
        else:
            return jsonify({'status': 0, 'message': '这个文件类型不支持！'})

        def get_result(future):
            success, message, precision = future.result()
            # 返回照片是否为伪造的，以及准确率
            if success:
                return jsonify({'status': 1, 'message': message, 'precision': precision})
            else:
                return jsonify({'status': 1, 'message': message, 'precision': precision})

        return get_result(future)
        # future.add_done_callback(get_result)
        # return jsonify({'status': 0, 'message': '文件正在处理，请稍后！'})
    else:
        return jsonify({'status': 0, 'message': '文件类型不支持！'})



# 在应用关闭时调用清理函数，确保所有临时文件都被删除
@app.teardown_appcontext
def teardown(exc=None):
    # 清空临时文件夹
    cleanup_folder(app.config['TEMP_FOLDER'])
    cleanup_folder(app.config['UPLOAD_FOLDER'])


if __name__ == '__main__':
    # print("API server running at http://127.0.0.1:5000/process")
    app.run(debug=True, port=5000)
