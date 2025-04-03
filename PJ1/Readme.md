# Project-1 of “Neural Network and Deep Learning”

# MLP-MNIST

```
├── data                     # MNIST数据集
├── utils.py                 # 载入数据 & 数据处理 & 保存、加载训练好的模型 & 可视化loss,accuracy,weight,weights distribution的辅助函数
├── layers.py                # 输入层 & 全连接层 & 激活函数
├── training-core.py         # 优化器 & 损失函数 & 训练神经网络
├── model.py                 # 构建MLP模型
├── evaluate.py              # 评估模型
├── accuracy.py              # 计算accuracy

├── activation_experiment.py # 对比不同activation function性能
├── optimizer_experiment.py  # 对比不同optimizer性能
├── main.py                  # 主函数，在终端运行的代码

├── best_model               # 存放训练好的模型和参数

```

# CNN-MNIST

```
├── dataset                  # MNIST数据集
    ├── mnist.py             # 载入数据 & 数据处理
├── layers.py                # 卷积层 & 池化层 & 激活函数 & 全连接层 & 损失函数
├── trainer.py               # 训练网络
├── IM2COL.py                # 辅助卷积计算
├── SimpleConvNet.py         # 构建一个简单的CNN模型（ conv - relu - pool - affine - relu - affine - softmax）
├── optimizer.py             # 优化器实现
├── plot.py                  # 可视化loss,accuracy，卷积核权重
├── main.py                  # 主函数，在终端运行的代码
```
