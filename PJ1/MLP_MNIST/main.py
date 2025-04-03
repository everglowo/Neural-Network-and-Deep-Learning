import numpy as np
from utils import DataHandler, load_model,save_model, Visualizer
from model import Model
from layers import FullyConnectedLayer, ActivationReLU, ActivationSoftmax, ActivationSigmoid, ActivationTanh,InputLayer
from training_core import SGD, SGDMomentum,RMSProp, Adam, CELoss, Trainer
from evaluate import Evaluator
from accuracy import Accuracy
import matplotlib.pyplot as plt 

np.random.seed(0)

# Create dataset
data_handler = DataHandler('./data')
X_train, y_train, X_test, y_test = data_handler.load_mnist()
X, y, X_val, y_val = data_handler.split_validation(X_train, y_train, val_ratio=0.2)  # 增加训练集和验证集的比例
# 新增数据增强（增强1倍）
X, y = data_handler.augment(X, y, factor=1)
X, X_test = data_handler.scale(X, X_test)  # 数据归一化
X_val = data_handler.scale(X_val)  # 验证集归一化


def create_model(n_hidden=[128, 64], optimizer_name='SGDMomentum',
                 learning_rate=0.001, decay=1e-4, momentum=0.9, l2_reg_weight=1e-4,
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

# # 粗调
# learning_rates = [0.0001,0.001,0.01]
# n_hiddens= [[64,64],[128,64]]
# l2_reg_weights = [0.001,0.01,0.1]
 

# # 精调
# learning_rates = [0.01,0.05]
# n_hiddens= [[128,64]]
# l2_reg_weights = [1e-4,1e-3]


# 最优参数  
learning_rates = [0.05]
n_hiddens = [[128,64]]
l2_reg_weights = [0.0001]

# 超参数
n_epoch = 10
batch_size = 128  
# 选择使用的优化器
use_sgd_momentum = True  
momentum = 0.9
best_val_acc = 0.0

for lr in learning_rates:
    for neurons in n_hiddens:
        for l2 in l2_reg_weights:
                print(f"\n### Training with lr={lr}, neurons={neurons}, l2={l2}")
                
                # Create model with the optimal hyperparameters
                model = create_model(n_hidden=neurons,
                                     optimizer_name='SGDMomentum',
                                     learning_rate=lr,
                                     decay=1e-4,
                                     momentum=0.9,
                                     l2_reg_weight=l2,
                                     activation=['relu', 'relu'], input_dim=784
                                     )
                # Train the model
                trainer = Trainer(model)
                trainer.train(X, y, epochs=n_epoch, batch_size=batch_size,
                              print_every=200, val_data=(X_val, y_val))
                model.set_parameters(model.best_weights)
                # Save the parameters and the model
                model.save_params('./best_model/best_mnist_params.pkl')
                save_model(model, './best_model/best_mnist_model.pkl')
                print("### Model saved.")
                # Plot loss, accuracy and learning rate
                trainer.plot_figs()
                    
                # 可视化权重
                Visualizer.visualize_weights(model.best_weights[0][0], layer_name=f"Hidden Layer {1}")
                    
                # 可视化偏置
                Visualizer.visualize_biases(model.best_weights[0][1], layer_name=f"Hidden Layer {1}")

                # 可视化权重
                Visualizer.visualize_weights(model.best_weights[1][0], layer_name=f"Hidden Layer {2}")
                    
                # 可视化偏置
                Visualizer.visualize_biases(model.best_weights[1][1], layer_name=f"Hidden Layer {2}")
                # Plot weights
                Visualizer.visualize_first_fc_layer(model)

                # Plot weights
                Visualizer.visualize_second_fc_layer(model)



# 测试最优模型
# model = load_model('./best_model/best_mnist_model.pkl')
# evaluator = Evaluator(model)
# evaluator.eval(X_test, y_test, batch_size=128)
                

