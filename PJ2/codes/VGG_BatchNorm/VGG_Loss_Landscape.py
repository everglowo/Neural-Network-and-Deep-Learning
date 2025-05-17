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


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=20, duration=30):
    model.to(device)
    # max_val_accuracy = 0
    loss_values = []  # use this to record the loss value of each step / duration
    grad_distances = []  # use this to record the loss gradient distances
    beta_values = []  # use this to record the beta smoothness values
    iteration = 0  # count the total number of iterations
    accumulated_loss = 0  # accumulate the loss values
    previous_grad = None  # store the gradient of the previous step
    previous_param = None  # store the parameters of the previous step

    for epoch in tqdm(range(epochs_n), unit='epoch', ncols=75):
        if scheduler is not None:
            scheduler.step()  # update the learning rate if scheduler is provided
        model.train()

        for data in train_loader:
            iteration += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            accumulated_loss += loss.item()

            if iteration % duration == 0:
                # train_acc = get_accuracy(model, train_loader)
                # val_acc = get_accuracy(model, val_loader, is_train=False)
                # max_val_accuracy = max(val_acc, max_val_accuracy)
                current_grad = model.classifier[-1].weight.grad.detach().clone()  # get the current gradient
                current_param = model.classifier[-1].weight.detach().clone()  # get the current parameters
                if previous_grad is not None:
                    grad_distance = torch.dist(current_grad, previous_grad).item()  # calculate the gradient distance
                    grad_distances.append(grad_distance)
                if previous_param is not None:
                    param_distance = torch.dist(current_param, previous_param).item()  # calculate the parameter distance
                    beta_values.append(grad_distance / (param_distance + 1e-3))  # calculate beta smoothness
                previous_grad = current_grad
                previous_param = current_param
                loss_values.append(accumulated_loss / duration)
                accumulated_loss = 0
    # print(f'Finish Training! Training accuracy: {train_acc}, Best validation accuracy: {max_val_accuracy}')
    print(f'Finish Training!')
    return loss_values, grad_distances, beta_values


# def plot_metric(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
#     min_vgg = np.min(list_vgg, axis=0)
#     max_vgg = np.max(list_vgg, axis=0)
#     steps = np.arange(0, len(min_vgg)) * duration
#     ax.plot(steps, min_vgg, 'g-', alpha=0.8, label=label_vgg)
#     ax.plot(steps, max_vgg, 'g-', alpha=0.8)
#     ax.fill_between(steps, min_vgg, max_vgg, color='g', alpha=0.4)
    
#     min_vgg_bn = np.min(list_vgg_bn, axis=0)
#     max_vgg_bn = np.max(list_vgg_bn, axis=0)
#     steps = np.arange(0, len(min_vgg_bn)) * duration
#     ax.plot(steps, min_vgg_bn, 'r', alpha=0.8, label=label_vgg_bn)
#     ax.plot(steps, max_vgg_bn, 'r', alpha=0.8)
#     ax.fill_between(steps, min_vgg_bn, max_vgg_bn, color='r', alpha=0.4)
    
#     ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
#     ax.legend()

def plot_loss_landscape(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    min_vgg = np.min(list_vgg, axis=0)
    max_vgg = np.max(list_vgg, axis=0)
    steps = np.arange(0, len(min_vgg)) * duration
    ax.plot(steps, min_vgg, 'g-', alpha=0.8, label=label_vgg)
    ax.plot(steps, max_vgg, 'g-', alpha=0.8)
    ax.fill_between(steps, min_vgg, max_vgg, color='g', alpha=0.4)
    
    min_vgg_bn = np.min(list_vgg_bn, axis=0)
    max_vgg_bn = np.max(list_vgg_bn, axis=0)
    steps = np.arange(0, len(min_vgg_bn)) * duration
    ax.plot(steps, min_vgg_bn, 'r', alpha=0.8, label=label_vgg_bn)
    ax.plot(steps, max_vgg_bn, 'r', alpha=0.8)
    ax.fill_between(steps, min_vgg_bn, max_vgg_bn, color='r', alpha=0.4)
    
    ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax.legend()


def plot_gradient_distance(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    min_vgg = np.min(list_vgg, axis=0)
    max_vgg = np.max(list_vgg, axis=0)
    steps = np.arange(0, len(min_vgg)) * duration
    ax.plot(steps, min_vgg, 'g-', alpha=0.8, label=label_vgg)
    ax.plot(steps, max_vgg, 'g-', alpha=0.8)
    ax.fill_between(steps, min_vgg, max_vgg, color='g', alpha=0.4)
    
    min_vgg_bn = np.min(list_vgg_bn, axis=0)
    max_vgg_bn = np.max(list_vgg_bn, axis=0)
    steps = np.arange(0, len(min_vgg_bn)) * duration
    ax.plot(steps, min_vgg_bn, 'r', alpha=0.8, label=label_vgg_bn)
    ax.plot(steps, max_vgg_bn, 'r', alpha=0.8)
    ax.fill_between(steps, min_vgg_bn, max_vgg_bn, color='r', alpha=0.4)
    
    ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax.legend()
    
def plot_beta_smoothness(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    max_vgg = np.max(np.asarray(list_vgg), axis=0)
    max_vgg_bn = np.max(np.asarray(list_vgg_bn), axis=0)
    steps = np.arange(0, len(max_vgg)) * duration
    ax.plot(steps, max_vgg, 'g-', alpha=0.8, label=label_vgg)
    ax.plot(steps, max_vgg_bn, 'r', alpha=0.8, label=label_vgg_bn)
    
    ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax.legend()


# Train your model
if __name__ == '__main__':
    epochs = 20
    learning_rates =[0.05,0.075,0.1,0.15]
    duration = 30  # record every step
    set_random_seeds(seed_value=2020, device=device)
    
    grad_list_vgg = []
    loss_list_vgg = []
    beta_list_vgg = []
    grad_list_vgg_bn = []
    loss_list_vgg_bn = []
    beta_list_vgg_bn = []
    
    for lr in learning_rates:
        print(f'----Training Standard VGG-A, learning rate: {lr}----')
        model_vgg = VGG_A()
        optimizer = torch.optim.SGD(model_vgg.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss_vals, grads, beta_vals = train(model_vgg, optimizer, criterion, train_loader, val_loader, epochs_n=epochs, duration=duration)
        grad_list_vgg.append(grads)
        loss_list_vgg.append(loss_vals)
        beta_list_vgg.append(beta_vals)
    
    for lr in learning_rates:
        print(f'----Training VGG-A with BatchNorm, learning rate: {lr}----')
        model_vgg_bn = VGG_A_BatchNorm()
        optimizer = torch.optim.SGD(model_vgg_bn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss_vals, grads, beta_vals = train(model_vgg_bn, optimizer, criterion, train_loader, val_loader, epochs_n=epochs, duration=duration)
        grad_list_vgg_bn.append(grads)
        loss_list_vgg_bn.append(loss_vals)
        beta_list_vgg_bn.append(beta_vals)
    
    
    plt.style.use('ggplot')
    # Plot Loss Landscape
    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_loss_landscape(ax, np.array(loss_list_vgg), np.array(loss_list_vgg_bn), 'Loss Landscape', 'Loss', 'Standard VGG', 'VGG with BatchNorm', duration)
    plt.savefig('./figs/Loss_Landscape.png')
    plt.close()
    
    # Plot Gradient Distance
    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_gradient_distance(ax, np.array(grad_list_vgg), np.array(grad_list_vgg_bn), 'Gradient Predictiveness', 'Gradient Distance', 'Standard VGG', 'VGG with BatchNorm', duration)
    plt.savefig('./figs/Gradient_Distance.png')
    plt.close()
    
    # Plot Beta Smoothness
    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_beta_smoothness(ax, np.array(beta_list_vgg), np.array(beta_list_vgg_bn), 'Beta Smoothness', 'Beta', 'Standard VGG', 'VGG with BatchNorm', duration)
    plt.savefig('./figs/Beta_Smoothness.png')
    plt.close()


    # plt.style.use('ggplot')
    # fig, axs = plt.subplots(1, 3, figsize=(27, 6), dpi=800)

    # plot_metric(axs[0], np.array(loss_list_vgg), np.array(loss_list_vgg_bn), 'Loss Landscape', 'Loss', 'Standard VGG', 'VGG with BatchNorm', duration)
    # plot_metric(axs[1], np.array(grad_list_vgg), np.array(grad_list_vgg_bn), 'Gradient Predictiveness', 'Gradient Distance', 'Standard VGG', 'VGG with BatchNorm', duration)
    # plot_metric(axs[2], np.array(beta_list_vgg), np.array(beta_list_vgg_bn), 'Beta Smoothness', 'Beta', 'Standard VGG', 'VGG with BatchNorm', duration)

    # plt.savefig('./figs/BN_analysis.png')
    # plt.close()