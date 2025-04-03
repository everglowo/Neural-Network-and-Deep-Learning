import numpy as np
import matplotlib.pyplot as plt
from utils import DataHandler
from model import Model
from layers import FullyConnectedLayer, ActivationReLU, ActivationSoftmax, InputLayer
from training_core import SGD, SGDMomentum, RMSProp, Adam, CELoss, Trainer
from evaluate import Evaluator
from accuracy import Accuracy

np.random.seed(0)
data_handler = DataHandler('./data')
X_train, y_train, X_test, y_test = data_handler.load_mnist()
X, y, X_val, y_val = data_handler.split_validation(X_train, y_train, val_ratio=0.2)
X, X_test = data_handler.scale(X, X_test)
X_val = data_handler.scale(X_val)

n_epoch = 10
batch_size = 128

def create_model(n_hidden=[128, 64], optimizer_name='SGD',
                 learning_rate=0.05, decay=1e-4, momentum=0.9, l2_reg_weight=1e-4):
    """
    构建可变隐藏层神经网络，并允许选择不同的优化器。

    参数：
        - n_hidden: list，每一层隐藏层的神经元数量，例如 [512, 256] 表示两层隐藏层
        - optimizer_name: str，优化器名称，可选 'SGD', 'SGDMomentum', 'RMSProp', 'Adam'
        - learning_rate: float，学习率
        - decay: float，学习率衰减
        - momentum: float，仅用于 SGDMomentum
        - l2_reg_weight: float，L2 正则化权重

    返回：
        - model: 训练好的 Model 对象
    """
    model = Model()
    model.add(InputLayer())

    # 添加隐藏层
    input_size = X.shape[1]  # 784 for MNIST
    for neurons in n_hidden:
        model.add(FullyConnectedLayer(input_size, neurons, l2_reg_weight))
        model.add(ActivationReLU())
        input_size = neurons  # 下一层的输入维度

    # 输出层
    model.add(FullyConnectedLayer(input_size, 10, l2_reg_weight))
    model.add(ActivationSoftmax())

    print(f"### Model architecture: {X.shape[1]} -> {' -> '.join(map(str, n_hidden))} -> 10")

    # 选择优化器
    optimizers = {
        'SGD': SGD(learning_rate, decay),
        'SGDMomentum': SGDMomentum(learning_rate, decay, momentum),
        'RMSProp': RMSProp(learning_rate, decay),
        'Adam': Adam(learning_rate, decay)
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized. Choose from {list(optimizers.keys())}")

    optimizer = optimizers[optimizer_name]

    # 设定损失函数 & 评估指标
    model.set_items(
        loss=CELoss(),
        optimizer=optimizer,
        accuracy=Accuracy()
    )
    model.finalize()
    
    return model

# 定义要比较的优化器
optimizers = {
    'SGD',
    'SGDMomentum',
    'RMSProp',
    'Adam'
}

# 存储结果
optimizer_results = {
    'train_loss': {},
    'val_loss': {},
    'val_acc': {}
}

# 训练并记录结果
for opt_name in optimizers:
    print(f"\n=== Training with {opt_name} ===")
    model = create_model(optimizer_name=opt_name)
    trainer = Trainer(model)
    trainer.train(X, y, epochs=n_epoch, 
                 batch_size=batch_size, val_data=(X_val, y_val))
    
    # 保存训练过程中的指标
    optimizer_results['train_loss'][opt_name] = trainer.loss_train
    optimizer_results['val_loss'][opt_name] = trainer.loss_val
    optimizer_results['val_acc'][opt_name] = trainer.acc_val

# 绘制优化器比较图
def plot_optimizer_comparison(results):
    plt.figure(figsize=(15, 5))
    
    # 训练损失对比
    plt.subplot(1, 2, 1)
    for opt_name, losses in results['train_loss'].items():
        plt.plot(losses, label=opt_name, linewidth=2)
    plt.title('Training Loss Comparison ')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 验证损失对比
    plt.subplot(1, 2, 2)
    for opt_name, losses in results['val_loss'].items():
        plt.plot(losses, label=opt_name, linewidth=2)
    plt.title('Validation Loss Comparison (ReLU+ReLU)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("pic has been saved as optimizer_comparison.png")

# 调用绘图函数
plot_optimizer_comparison(optimizer_results)