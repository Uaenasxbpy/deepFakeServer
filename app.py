# app
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from Moddel import test
# 处理跨域亲请求
app = Flask(__name__)
CORS(app)

# 设置上传文件夹的路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# TODO 处理图像文件的函数
def process_image(file):
    '''
    判断图片是否为伪造的
    :param file:
    :return:
    '''
    print(file)
    # test.detect_image(image=100)
    predicted_label, confidence = test.detect_image(image=file)
    return predicted_label, "照片是真的", "准确率为:"+str(confidence * 100) + "%"

# TODO 处理视频文件的函数
def process_video(file):
    '''
    判断视频是否为伪造的
    :param file:
    :return:
    '''
    return False, "视频是假的", "准确率高达95.35%"

# 检查文件名称是否可用
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# TODO API接口的路由
@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'status': 0, 'message': 'No file part'})
    file = request.files['file']
    # 构建完整的文件路径
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    print(filepath)
    file.save(filepath)
    # 文件名不存在
    if file.filename == '':
        return jsonify({'status': 0, 'message': 'No selected file'})

    # 文件名称可用
    if file and allowed_file(file.filename):
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # image
            success, message, precision = process_image(file)
        elif file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
            # video
            success, message, precision = process_video(file)
        else:
            # 文件类型错误
            return jsonify({'status': 0, 'message': 'Unsupported file type'})

        # 返回照片是否为伪造的，以及准确率
        if success:
            return jsonify({'status': 1, 'message': message, 'precision': precision})
        else:
            return jsonify({'status': 0, 'message': message, 'precision': precision})

    else:
        return jsonify({'status': 0, 'message': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
