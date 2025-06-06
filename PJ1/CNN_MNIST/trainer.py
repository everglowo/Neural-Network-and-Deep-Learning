import numpy as np
from optimizer import *
import numpy as np


class Trainer:
    """进行神经网络的训练的类
    """

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                                'adagrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []

        self.test_loss_list = []

        self.train_acc_list = []
        self.test_acc_list = []

        self.verbose_interval = 100  # 设定每 10 个步骤输出一次损失
        self.step_count = 0  # 记录训练步骤

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        train_loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(train_loss)

        # 计算 test loss
        test_batch_mask = np.random.choice(len(self.x_test), self.batch_size)  
        x_test_batch = self.x_test[test_batch_mask]
        t_test_batch = self.t_test[test_batch_mask]
        test_loss = self.network.loss(x_test_batch, t_test_batch)
        self.test_loss_list.append(test_loss)
        
        self.step_count += 1
        if self.step_count % self.verbose_interval == 0:
            if self.verbose: 
                print(f"Step {self.step_count}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(f"=== epoch: {self.current_epoch}, Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f} ===")

        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))