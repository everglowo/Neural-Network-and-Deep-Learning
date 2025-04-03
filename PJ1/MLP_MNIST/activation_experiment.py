import numpy as np
import matplotlib.pyplot as plt
from utils import DataHandler
from model import Model
from layers import FullyConnectedLayer, ActivationReLU, ActivationSigmoid, ActivationTanh, ActivationSoftmax, InputLayer
from training_core import SGD, SGDMomentum,RMSProp,Adam, CELoss, Trainer
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


def create_model(n_hidden=[128, 64], optimizer_name='SGDMomentum',
                 learning_rate=0.005, decay=1e-3, momentum=0.9, l2_reg_weight=1e-3,
                 activation=['relu', 'relu'], input_dim=784):
    """
    构建可变隐藏层神经网络，允许选择优化器和激活函数。

    参数：
        - n_hidden: list，每一层隐藏层的神经元数量，例如 [512, 256] 表示两层隐藏层
        - optimizer_name: str，优化器名称，可选 'SGD', 'SGDMomentum', 'RMSProp', 'Adam'
        - learning_rate: float，学习率
        - decay: float，学习率衰减
        - momentum: float，仅用于 SGDMomentum
        - l2_reg_weight: float，L2 正则化权重
        - activation: list/str，隐藏层的激活函数，长度需与n_hidden一致，或统一使用某激活函数
        - input_dim: int，输入数据的特征维度（如MNIST为784）

    返回：
        - model: 构建好的 Model 对象
    """
    model = Model()
    model.add(InputLayer())

    # 处理激活函数参数
    if activation is None:
        activation = ['relu'] * len(n_hidden)
    elif isinstance(activation, str):
        activation = [activation] * len(n_hidden)
    else:
        if len(activation) != len(n_hidden):
            raise ValueError("activation length should correspond with n_hidden")

    # 添加隐藏层
    input_size = input_dim
    for neurons, activ in zip(n_hidden, activation):
        model.add(FullyConnectedLayer(input_size, neurons, l2_reg_weight))
        # 根据激活函数名称添加对应层
        activ = activ.lower()
        if activ == 'relu':
            model.add(ActivationReLU())
        elif activ == 'sigmoid':
            model.add(ActivationSigmoid())
        elif activ == 'tanh':
            model.add(ActivationTanh())

        input_size = neurons

    # 输出层
    model.add(FullyConnectedLayer(input_size, 10, l2_reg_weight))
    model.add(ActivationSoftmax())

    # 打印模型结构（含激活函数）
    arch = [str(input_dim)]
    for n, a in zip(n_hidden, activation):
        arch.append(f"{n} ({a})")
    arch.append("10")
    print(f"### Model Structure: {' -> '.join(arch)}")

    # 选择优化器
    optimizers = {
        'SGD': SGD(learning_rate, decay),
        'SGDMomentum': SGDMomentum(learning_rate, decay, momentum),
        'RMSProp': RMSProp(learning_rate, decay),
        'Adam': Adam(learning_rate, decay)
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"优化器 '{optimizer_name}' 不存在，可选：{list(optimizers.keys())}")
    optimizer = optimizers[optimizer_name]

    # 配置模型
    model.set_items(
        loss=CELoss(),
        optimizer=optimizer,
        accuracy=Accuracy()
    )
    model.finalize()
    
    return model


activation_functions = ['relu','sigmoid', 'tanh']
results_acc={}
results_loss = {}
all_losses = {}
all_accuracies = {}

for act_fn1 in activation_functions:
    for act_fn2 in activation_functions:
        model = create_model( activation=[act_fn1, act_fn2])
        trainer = Trainer(model)
        # 调用train方法进行训练
        trainer.train(X, y, epochs=n_epoch, batch_size=batch_size, val_data=(X_val, y_val))
        model.set_parameters(model.best_weights)
        evaluator = Evaluator(model)
        acc_test, loss_test = evaluator.eval(X_test, y_test, batch_size=128)
        results_acc[f"{act_fn1} + {act_fn2}"] = acc_test
        results_loss[f"{act_fn1} + {act_fn2}"] = loss_test
        all_losses[f"{act_fn1} + {act_fn2}"] = trainer.loss_val
        all_accuracies[f"{act_fn1} + {act_fn2}"] = trainer.acc_val

def plot_activation_comparison(all_losses, all_accuracies):
    import os
    if not os.path.exists('activation_plots'):
        os.makedirs('activation_plots')
    
    # 使用当前可用的样式替代 'seaborn'
    available_styles = plt.style.available
    preferred_styles = ['seaborn-v0_8', 'ggplot', 'seaborn', 'default']
    selected_style = next((style for style in preferred_styles if style in available_styles), 'default')
    plt.style.use(selected_style)
    
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.rcParams['font.size'] = 12
    
    for combo_name in all_losses.keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 绘制损失曲线
        ax1.plot(all_losses[combo_name], 'b-', label='Training Loss')
        ax1.set_title(f'{combo_name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 绘制准确率曲线
        ax2.plot(all_accuracies[combo_name], 'r-', label='Validation Accuracy')
        ax2.set_title(f'{combo_name} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f'activation_plots/{combo_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"plots haven been saved in activation_plots")

# 调用绘图函数
plot_activation_comparison(all_losses, all_accuracies)
