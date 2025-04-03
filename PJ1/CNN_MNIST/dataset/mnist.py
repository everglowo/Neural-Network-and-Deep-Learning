# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np
import cv2
import random

url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # mirror site
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

# 图像增强类
class ImageAugmentation:
    def __init__(self, rotation_range=15, translation_range=0.1, scale_range=(0.8, 1.2)):
        self.rotation_range = rotation_range  # 旋转角度范围
        self.translation_range = translation_range  # 平移范围
        self.scale_range = scale_range  # 缩放比例范围
    
    def rotate(self, image):
        """ 随机旋转图像 """
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)
        return rotated_image

    def translate(self, image):
        """ 随机平移图像 """
        rows, cols = image.shape
        tx = random.uniform(-self.translation_range, self.translation_range) * cols
        ty = random.uniform(-self.translation_range, self.translation_range) * rows
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)
        return translated_image

    def scale(self, image):
        """ 随机缩放图像 """
        rows, cols = image.shape
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # 如果缩放后尺寸不为28x28，进行裁剪或填充
        if resized_image.shape[0] != rows or resized_image.shape[1] != cols:
            resized_image = cv2.resize(resized_image, (cols, rows), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def augment(self, image):
        """ 对图像进行多个增强变换 """
        image = self.rotate(image)
        image = self.translate(image)
        image = self.scale(image)
        return image
# 加载 MNIST 数据，并支持数据增强
def load_mnist(normalize=True, flatten=True, one_hot_label=False, augment=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 是否将标签转换为one-hot数组
    flatten : 是否将图像展开为一维数组
    augment : 是否进行图像增强
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    
    if augment:
        augmenter = ImageAugmentation(rotation_range=10, translation_range=0.1, scale_range=(0.8, 1.2))
        augmented_train_images = []
        
        # 对训练集进行增强
        for img in dataset['train_img']:
            augmented_img = augmenter.augment(img[0])  # 取出单通道图像
            augmented_train_images.append(augmented_img)
        
        augmented_train_images = np.array(augmented_train_images)

        # 修正: 增加通道维度 (N, 28, 28) -> (N, 1, 28, 28)
        augmented_train_images = np.expand_dims(augmented_train_images, axis=1)

        dataset['train_img'] = np.concatenate([dataset['train_img'], augmented_train_images], axis=0)
        dataset['train_label'] = np.concatenate([dataset['train_label'], dataset['train_label']], axis=0)

        # print(f"Training data shape: {dataset['train_img']}, {dataset['train_label']}")
        # print(f"Test data shape: {dataset['test_img']}, {dataset['test_label']}")
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

