import numpy as np
import matplotlib.pyplot as plt
from SimpleConvNet import SimpleConvNet
from dataset.mnist import load_mnist
from trainer import Trainer
from plot import plot_loss_curve, plot_accuracy_curve,visualize_kernels

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False,augment=False)

max_epochs = 3

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)

trainer.train()

# 保存参数
network.save_params("params.pkl")
print("Saved Network Parameters!")


plot_loss_curve(trainer.train_loss_list, trainer.test_loss_list)
plot_accuracy_curve(trainer.train_acc_list, trainer.test_acc_list)
layer_name = "Conv1"  # 你需要用实际的层名替换这里的 "conv1"
visualize_kernels(network, layer_name)