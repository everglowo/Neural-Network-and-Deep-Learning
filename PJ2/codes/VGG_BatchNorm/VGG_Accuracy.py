import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
# device_id = [0,1,2,3]  # We only have one GPU
num_workers = 4
batch_size = 128
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# add our package dir to path 
# module_path = os.path.dirname(os.getcwd())
# home_path = module_path
# figures_path = os.path.join(home_path, 'reports', 'figures')
# models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
# device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X, y in train_loader:
    try:
        # print(f"Sample input: {X[0]}")
        # print(f"Sample label: {y[0]}")
        # print(f"Input shape: {X[0].shape}")
        img = np.transpose(X[0].cpu().numpy(), (1, 2, 0))
        # plt.imshow(img * 0.5 + 0.5)
        # plt.savefig('sample.png')
    except Exception as e:
        print("Error! Please check the DataLoader.")
        print(e)
    else:
        print("Data have been successfully loaded.")
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, is_train=True):
    if is_train == False:
        model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    if not is_train:
        # print(f"Validation Accuracy: {accuracy:.4f} \n")
        model.train() 
    # else:
        # print(f"Train Accuracy: {accuracy:.4f} \n")
    return accuracy

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire training process. 
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, duration=30, best_model_path=None):
    model.to(device)
    train_accuracy_curve = []
    val_accuracy_curve = []
    max_val_accuracy = 0
    iteration = 0

    for epoch in tqdm(range(epochs_n), unit='epoch', ncols=75):
        if scheduler is not None:
            scheduler.step()
        model.train()
        
        for data in train_loader:
            iteration += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iteration % duration == 0:
                train_acc = get_accuracy(model, train_loader)
                val_acc = get_accuracy(model, val_loader, is_train=False)
                train_accuracy_curve.append(train_acc)
                val_accuracy_curve.append(val_acc)
                max_val_accuracy = max(val_acc, max_val_accuracy)

    print(f'Finish Training! Best validation accuracy: {max_val_accuracy}')
    return train_accuracy_curve, val_accuracy_curve


def plot_accuracy(train_acc, val_acc, model_name, save_path, color, duration=1):
        steps = np.arange(0, len(train_acc)) * duration
        plt.style.use('ggplot')
        plt.figure(figsize=(7, 5), dpi=800)
        plt.plot(steps, train_acc, color=f'{color}', linestyle='-', label=f'Training: {model_name}', linewidth=1)
        plt.plot(steps, val_acc, color=f'{color}', linestyle=':', label=f'Validation: {model_name}', linewidth=1)
        plt.title(f'Accuracy Curve ({model_name})')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

# Train your model
if __name__ == '__main__':
    epo = 20
    lr = 0.1
    duration = 100
    set_random_seeds(seed_value=2024, device=device)
    
    print(f'----Training Standard VGG-A, learning rate: {lr}----')
    model_vgg = VGG_A()
    Opt = torch.optim.SGD(model_vgg.parameters(), lr=lr)
    Loss = nn.CrossEntropyLoss()
    tr_acc_vgg, val_acc_vgg = train(
        model_vgg, Opt, Loss, train_loader, val_loader, epochs_n=epo, duration=duration)

    print(f'----Training VGG-A with BatchNorm, learning rate: {lr}----')
    model_bn = VGG_A_BatchNorm()
    Opt = torch.optim.SGD(model_bn.parameters(), lr=lr)
    Loss = nn.CrossEntropyLoss()
    tr_acc_bn, val_acc_bn = train(
        model_bn, Opt, Loss, train_loader, val_loader, epochs_n=epo, duration=duration)
    
    # plot_accuracy(tr_acc_vgg, val_acc_vgg, 'Standard VGG', './figs/Standard_VGG_acc.png', "g", duration)
    # plot_accuracy(tr_acc_bn, val_acc_bn, 'VGG with BatchNorm', './figs/BatchNorm_VGG_acc.png', "r", duration)

    steps = np.arange(0, len(tr_acc_vgg)) * duration
    plt.style.use('ggplot')
    plt.figure(figsize=(7, 5), dpi=800)
    plt.plot(steps, tr_acc_vgg, 'g-', label='Training: Standard VGG', linewidth=1.5)
    plt.plot(steps, val_acc_vgg, 'g:', label='Validation: Standard VGG', linewidth=1.5)
    plt.plot(steps, tr_acc_bn, 'r-', label='Training: VGG with BatchNorm', linewidth=1.5)
    plt.plot(steps, val_acc_bn, 'r:', label='Validation: VGG with BatchNorm', linewidth=1.5)
    plt.title('Accuracy Curve Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./figs/Comparison_acc.png')
    plt.close()