# 对文件夹中的照片按照相似度排序并重命名
import os
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, RepeatVector, Attention # type: ignore
from tensorflow.keras.models import Model # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 加载预训练的 VGG16 模型，不包含顶层全连接层
base_model = VGG16(weights='imagenet', include_top=False)

# 添加自定义顶层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = RepeatVector(1)(x)  # 将二维张量转换为三维张量
x = Attention()([x, x])  # 添加注意力机制
predictions = Dense(1000, activation='softmax')(x)

# 构建新模型
model = Model(inputs=base_model.input, outputs=predictions)

# 冻结预训练层的权重
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy')

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