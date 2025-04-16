import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curve(train_loss_list, test_loss_list):
    """绘制 loss 曲线"""
    x = np.arange(len(train_loss_list))  # 获取步数作为x轴
    plt.figure(figsize=(10, 5))
    plt.plot(x, train_loss_list, label='Train Loss', color='blue')
    plt.plot(x, test_loss_list, label='Test Loss', color='red')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

def plot_accuracy_curve(train_acc_list, test_acc_list):
    """绘制 accuracy 曲线"""
    epochs = np.arange(1, len(train_acc_list) + 1)  # 从1开始，表示epoch数
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc_list, marker='o', label='Train Accuracy', color='blue')
    plt.plot(epochs, test_acc_list, marker='s', label='Test Accuracy', color='red')
    plt.xlabel("Epoch")  # 修改x轴标签为"Epoch"
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)  # 保证accuracy在0到1之间
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_kernels(model, layer_name):
    """可视化卷积核"""
    weights = model.params['W1']
    num_kernels = weights.shape[0]
    rows = int(np.sqrt(num_kernels))
    cols = int(np.ceil(num_kernels / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            kernel = weights[i, 0]  # 只取单通道
            ax.imshow(kernel, cmap='gray')
            ax.axis('off')
    plt.suptitle(f"Visualization of {layer_name} Kernels")
    plt.show()
