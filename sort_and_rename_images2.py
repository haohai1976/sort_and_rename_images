# 对文件夹中的照片按照相似度排序并重命名
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 加载预训练的 ResNet50 模型，不包含顶层全连接层
base_model = ResNet50(weights='imagenet', include_top=False)

# 添加自定义顶层
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(1000, activation='softmax')(x)

# 构建新模型
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# 冻结预训练层的权重
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy')

# 定义 FGSM 生成对抗样本的函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = tf.sign(data_grad)
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    return perturbed_image

# 定义训练步骤
def train_step(images, labels, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    
    gradients = tape.gradient(loss, images)
    perturbed_images = fgsm_attack(images, epsilon, gradients)
    
    with tf.GradientTape() as tape:
        tape.watch(perturbed_images)
        predictions = model(perturbed_images, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 定义优化器
optimizer = tf.keras.optimizers.Adam()

# 训练模型
epsilon = 0.01
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in zip(x_train, y_train):
        images = np.expand_dims(images, axis=0)
        images = tf.convert_to_tensor(images) # 将 NumPy 数组转换为 TensorFlow 张量
        labels = np.expand_dims(labels, axis=0)
        labels = tf.one_hot(labels, depth=1000) # 将 labels 转换为 one-hot 编码
        labels = tf.reduce_sum(labels, axis=1)  # 假设每个样本只有一个标签
        train_step(images, labels, epsilon)

# 选择文件夹
def select_folder():
    # 创建Tkinter根窗口并隐藏
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 打开文件夹选择对话框并返回选中的文件夹路径
    folder_path = filedialog.askdirectory()  # 打开文件夹选择对话框
    return folder_path

# 提取图像特征
def extract_features(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data)
        features = features.flatten()
        return features
    except Exception as e:
        print(f"错误: {e}")
        return None

def sort_images_by_similarity(folder_path):
    features = []
    filenames = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        feature = extract_features(image_path)
        if feature is not None:
            features.append(feature)
            filenames.append(filename)
    
    if len(features) == 0:
        return []
    
    features = np.array(features)
    similarity_matrix = cosine_similarity(features)
    sorted_indices = np.argsort(similarity_matrix.sum(axis=1))[::-1]
    sorted_filenames = [filenames[i] for i in sorted_indices]
    return sorted_filenames


def rename_images(folder_path, sorted_image_files):
    # 重命名图像文件
    for i, image_file in enumerate(sorted_image_files):
        old_path = os.path.join(folder_path, image_file)
        new_name = f"{i+1:03d}_{image_file}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

# 主进程
if __name__ == "__main__":
    folder_path = select_folder()
    sorted_image_files = sort_images_by_similarity(folder_path)
    rename_images(folder_path, sorted_image_files)


    