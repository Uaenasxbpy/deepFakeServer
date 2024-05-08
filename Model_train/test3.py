import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_dataset(data_dir):
    print("Loading dataset from:", data_dir)
    images = []
    labels = []

    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            print("Processing category:", category)
            for image_file in os.listdir(category_dir):
                image_path = os.path.join(category_dir, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                images.append(image)

                # 检查文件名格式并提取方法标签
                if category == 'fake':  # 不是 "real" 类别的图像

                    labels.append('fake')
                else:  # "real" 类别的图像
                    labels.append('real')



    print("Dataset loading complete.")
    return np.array(images), np.array(labels)

# 加载数据集
train_images, train_labels = load_dataset("data/train")
test_images, test_labels = load_dataset("data/valid")

# 将标签转换为数字编码
label_map = {'true': 0, 'fake': 1}
train_labels = np.array([label_map.get(label, 0) for label in train_labels])
test_labels = np.array([label_map.get(label, 0) for label in test_labels])

# 将数据集拆分为训练集和验证集
print("Splitting dataset into train and validation sets...")
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# 归一化图像数据
print("Normalizing image data...")
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# 构建模型
print("Building model...")
def build_model():
    # 图像数据输入
    image_input = tf.keras.layers.Input(shape=(224, 224, 3), name='image_input')
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(pool2)
    flat1 = tf.keras.layers.Flatten()(conv3)

    # 全连接层
    dense1 = tf.keras.layers.Dense(64, activation='relu')(flat1)
    output = tf.keras.layers.Dense(2, activation='softmax')(dense1)  # 2个类别：true, deepfake, face2face, faceswap

    # 创建模型
    model = tf.keras.models.Model(inputs=image_input, outputs=output)
    return model

# 创建模型实例
model = build_model()

# 打印模型概况
print("Model summary:")
model.summary()

# 编译模型
print("Compiling model...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
print("Training model...")
history = model.fit(train_images, train_labels, epochs=60, batch_size=32, validation_data=(val_images, val_labels))

# 可视化训练过程
print("Visualizing training history...")
import matplotlib.pyplot as plt

def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

plot_training(history)

# 评估模型
print("Evaluating model on test data...")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

# 保存模型
model.save("your_model.h5")
print("Model saved successfully.")