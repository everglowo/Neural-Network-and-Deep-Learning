# This file contains the utility functions for the project.
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import os
from scipy.ndimage import rotate, shift, zoom

def mnist_reader(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from path"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels



class DataHandler:
    '''Class for loading and preprocessing data.'''
    
    def __init__(self, path=None):
        self.path = path
    
    # Loads a MNIST dataset
    def load_mnist(self, path=None):
        if path is None:
            path = self.path
        X_train, y_train = mnist_reader(path, kind='train')
        X_test, y_test = mnist_reader(path, kind='t10k')

        return X_train, y_train, X_test, y_test
    
    # Shuffle the training dataset
    def shuffle(self, X, y):
        keys = np.arange(X.shape[0])
        np.random.shuffle(keys)
        return X[keys], y[keys]

    # Scale samples to [-1, 1]
    def scale(self, X, X_test=None):
        if X_test is None:
            return (X - 127.5) / 127.5
        else:
            X = (X - 127.5) / 127.5
            X_test = (X_test - 127.5) / 127.5
            return X, X_test
    
    # Split a portion of the training set for validation
    def split_validation(self, X_train, y_train, val_ratio=0.25):
        total_size = X_train.shape[0]
        validation_size = int(total_size * val_ratio)
        # Shuffle the dataset before splitting
        X_train, y_train = self.shuffle(X_train, y_train)
        X_val = X_train[:validation_size]
        y_val = y_train[:validation_size]
        X_train_new = X_train[validation_size:]
        y_train_new = y_train[validation_size:]
        return X_train_new, y_train_new, X_val, y_val
    
        # 新增数据增强方法（保持原有方法不变）
    def augment(self, X, y, factor=1):
        """
        简单数据增强
        :param X: 原始图像数据 (n_samples, 784)
        :param y: 对应标签
        :param factor: 增强倍数（每个样本生成的新样本数）
        :return: 增强后的数据 (X_augmented, y_augmented)
        """
        X_aug = [X]
        y_aug = [y]
        
        for _ in range(factor):
            transformed = []
            for img in X:
                # 随机选择一种变换
                choice = np.random.choice(['shift', 'rotate', 'zoom'])
                img_2d = img.reshape(28, 28)
                
                if choice == 'shift':
                    # 随机平移（最大2像素）
                    dx, dy = np.random.randint(-2, 3, size=2)
                    transformed_img = shift(img_2d, [dy, dx], mode='constant')
                elif choice == 'rotate':
                    # 随机旋转（-15~15度）
                    angle = np.random.uniform(-15, 15)
                    transformed_img = rotate(img_2d, angle, mode='nearest', reshape=False)
                elif choice == 'zoom':
                    # 随机缩放（0.9~1.1倍）
                    scale = np.random.uniform(0.9, 1.1)
                    zoomed = zoom(img_2d, scale)
                    
                    # 裁剪/填充回28x28
                    h, w = zoomed.shape
                    if h > 28:
                        start = (h - 28) // 2
                        transformed_img = zoomed[start:start+28, :]
                    else:
                        pad = (28 - h) // 2
                        transformed_img = np.pad(zoomed, [(pad, 28-h-pad), (0,0)], mode='constant')
                        
                    if w > 28:
                        start = (w - 28) // 2
                        transformed_img = transformed_img[:, start:start+28]
                    else:
                        pad = (28 - w) // 2
                        transformed_img = np.pad(transformed_img, [(0,0), (pad, 28-w-pad)], mode='constant')
                
                transformed.append(transformed_img.ravel())
            
            X_aug.append(np.array(transformed))
            y_aug.append(y)
        
        return np.vstack(X_aug), np.concatenate(y_aug)


def save_model(model, path):
    '''Save a model to a file.'''
    model = copy.deepcopy(model)
    # Clear properties
    model.loss.reset_cum_loss()
    model.accuracy.reset_cum()
    model.input_layer.__dict__.pop('output', None)
    model.loss.__dict__.pop('dinputs', None)
    for layer in model.layers:
        for property in ['inputs', 'output', 'dinputs',
                         'dweights', 'dbiases']:
            layer.__dict__.pop(property, None)
    
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load_model(path):
    '''Load a model from a file.'''
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model




class Visualizer:
    '''Class for visualization.'''
    
    @staticmethod
    def plot_loss(train_loss, val_loss=None):
        train_steps = range(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_loss, label='Train Loss')
        if val_loss:
            epoch_length = len(train_loss) // len(val_loss)
            val_steps = range(epoch_length, len(train_loss) + 1, epoch_length)
            plt.plot(val_steps, val_loss, label='Validation Loss',
                     linestyle='-', marker='o')
        plt.title('Loss over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_accuracy(train_accuracy, val_accuracy=None):
        train_steps = range(1, len(train_accuracy) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_accuracy, label='Train Accuracy')
        if val_accuracy:
            epoch_length = len(train_accuracy) // len(val_accuracy)
            val_steps = range(epoch_length, len(
                train_accuracy) + 1, epoch_length)
            plt.plot(val_steps, val_accuracy,
                     label='Validation Accuracy', linestyle='-', marker='o')
        plt.title('Accuracy over Steps')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_learning_rate(lr_history):
        steps = range(1, len(lr_history) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(steps, lr_history, label='Learning Rate')
        plt.title('Learning Rate over steps')
        plt.xlabel('step')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.show()

    # @staticmethod
    # def visualize_weights(weights):
    #     num_neurons = weights.shape[1]

    #     input_dim = weights.shape[0]
    #     side_length = int(np.sqrt(input_dim))

    #     fig, axes = plt.subplots(8, 16, figsize=(16, 8))
    #     for i, ax in enumerate(axes.flat):
    #         if i < num_neurons:
    #             image = weights[:, i].reshape(side_length, side_length)
    #             ax.imshow(image, cmap='viridis', aspect='auto',
    #                       interpolation='nearest')
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #         else:
    #             ax.axis('off')

    #     plt.suptitle('Visualization of the First Layer Weights')
    #     plt.show()    
    @staticmethod
    def visualize_weights(weights, layer_name="Layer"):
        """可视化指定层的权重（兼容不同层结构）"""
        num_neurons = weights.shape[1]
        input_dim = weights.shape[0]

        # --- 固定第一层为8x16布局，第二层为8x8布局 ---
        if num_neurons == 128:  # 第一层
            rows, cols = 8, 16
        elif num_neurons == 64:  # 第二层
            rows, cols = 8, 8
        else:
            raise ValueError(f"Can not support the number: {num_neurons}")

        # --- 重塑权重为图像 ---
        side_length = int(np.sqrt(input_dim))
        if side_length**2 != input_dim:
            side_length = int(np.sqrt(input_dim))  # 非平方数时截断（如128→11x11不够，但保证可运行）

        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        axes = axes.flatten()
        
        for i in range(num_neurons):
            try:
                # 重塑权重为正方形（兼容MNIST输入）
                img = weights[:, i][:side_length**2].reshape(side_length, side_length)
                axes[i].imshow(img, cmap='viridis', aspect='auto', interpolation='nearest')
            except:
                axes[i].text(0.5, 0.5, f'Neuron {i}', ha='center', va='center')
            axes[i].axis('off')

        # 关闭多余子图
        for j in range(num_neurons, rows*cols):
            axes[j].axis('off')

        plt.suptitle(f'Visualization of the {layer_name} Layer Weights')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_biases(biases, layer_name="Layer"):
        """可视化偏置分布"""
        plt.figure(figsize=(8, 4))
        plt.hist(biases.flatten(), bins=20, color='steelblue', edgecolor='black')
        plt.title(f'{layer_name} Biases Distribution')
        plt.xlabel('Bias Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    @staticmethod
    def visualize_first_fc_layer(model):
        # 提取第一个全连接层（索引0）
        fc_layer = model.layers[1]  # layers[0]是InputLayer，layers[1]是第一层FC
        
 
        weights = fc_layer.weights
        
        # 创建画布
        plt.figure(figsize=(15, 6))
        
        # ------------------
        # 子图1：权重热力图
        # ------------------
        plt.subplot(1, 2, 1)
        
        # 使用对称的颜色范围（以0为中心）
        vmax = np.percentile(np.abs(weights), 99)  # 排除1%的极端值
        heatmap = plt.imshow(weights.T,  # 转置矩阵使输入维度在x轴
                            cmap='coolwarm',
                            aspect='auto',
                            vmin=-vmax,
                            vmax=vmax)
        
        plt.colorbar(heatmap, label='Weight Value')
        plt.title('First FC Layer Weights \nColor Range: ±{:.3f}'.format(vmax))
        plt.xlabel('Input Pixels ')
        plt.ylabel('Hidden Neurons')
        
        # ------------------
        # 子图2：权重分布
        # ------------------
        plt.subplot(1, 2, 2)
        
        # 计算统计指标
        mean = weights.mean()
        std = weights.std()
        
        # 绘制直方图（排除极端值）
        n, bins, patches = plt.hist(weights.flatten(), 
                                bins=100, 
                                range=(-vmax, vmax),
                                color='teal',
                                edgecolor='white',
                                alpha=0.7)
        
        # 添加统计标注
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
        plt.axvline(mean + std, color='orange', linestyle=':', label=f'±1σ ({std:.4f})')
        plt.axvline(mean - std, color='orange', linestyle=':')
        
        plt.title('Weight Distribution\nL2 Reg: {}'.format(fc_layer.l2_reg_weight))
        plt.xlabel('Weight Value')
        plt.ylabel('Count (log scale)')
        plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
   
    @staticmethod
    def visualize_second_fc_layer(model):
    # 提取第二个全连接层（注意层索引可能需要调整）
    # 根据你的模型结构，假设 layers[3] 是第二个FC层（索引从0开始）
    # InputLayer → FC1 → ReLU → FC2 → ReLU → FC3 → Softmax
        fc_layer = model.layers[3]
        
        weights = fc_layer.weights
        
        plt.figure(figsize=(16, 6))
        
        # ===================================================================
        # 子图1：权重矩阵可视化
        # ===================================================================
        plt.subplot(1, 2, 1)
        
        # 使用动态范围（排除极端值）
        abs_max = np.percentile(np.abs(weights), 99.9)
        matrix_view = weights[:256, :256]  # 仅显示前256x256区域
        
        im = plt.imshow(matrix_view, 
                    cmap='PiYG',  # 改用双色系增强对比
                    aspect='equal', 
                    vmin=-abs_max,
                    vmax=abs_max)
        
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"FC Layer 2 Weight Submatrix\n(First 256x256 of 512x512)\nTotal Range: [{weights.min():.3f}, {weights.max():.3f}]")
        plt.xlabel("Input Neurons (Layer 1 Output)")
        plt.ylabel("Output Neurons (Layer 2)")
        
        # ===================================================================
        # 子图2：权重分布与稀疏性分析
        # ===================================================================
        plt.subplot(1, 2, 2)
        
        # 计算统计量
        positive_ratio = (weights > 0).mean() * 100
        dead_neurons = (np.abs(weights).sum(axis=0) == 0).sum()  # 列和为0的神经元
        
        # 双坐标轴分析
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # 主坐标：分布直方图
        n, bins, _ = ax1.hist(weights.flatten(), 
                            bins=200, 
                            range=(-abs_max, abs_max),
                            color='darkcyan',
                            alpha=0.6,
                            log=True)
        
        # 副坐标：累积分布
        cumulative = np.cumsum(n) / np.sum(n)
        ax2.plot(bins[1:], cumulative, 'r--', linewidth=2)
        
        # 标注关键阈值
        for percentile in [25, 50, 75, 95]:
            value = np.percentile(weights, percentile)
            ax1.axvline(value, color='grey', linestyle=':', alpha=0.5)
            ax1.text(value, n.max()*0.8, f'{percentile}%', rotation=90, ha='right')
        
        ax1.set_title(
            f"Distribution | L2: {fc_layer.l2_reg_weight}\n"
            f"Pos/Neg Ratio: {positive_ratio:.1f}% | Dead: {dead_neurons} neurons"
        )
        ax1.set_xlabel("Weight Value")
        ax1.set_ylabel("Count (log)", color='darkcyan')
        ax2.set_ylabel("Cumulative %", color='red')
        
        plt.tight_layout()
        plt.show()
