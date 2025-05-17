import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, filter_multiplier=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes * filter_multiplier, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * filter_multiplier)
        self.conv2 = nn.Conv2d(
            planes * filter_multiplier, planes * filter_multiplier, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * filter_multiplier)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion * filter_multiplier:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion * filter_multiplier,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion * filter_multiplier)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, filter_multiplier=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, filter_multiplier=filter_multiplier)
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, filter_multiplier=filter_multiplier)
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, filter_multiplier=filter_multiplier)
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, filter_multiplier=filter_multiplier)
        self.linear = nn.Linear(
            512 * block.expansion * filter_multiplier, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, filter_multiplier):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                          stride, filter_multiplier))
            self.in_planes = planes * block.expansion * filter_multiplier
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_filtermul(filter_multiplier=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], filter_multiplier=filter_multiplier)


# Example usage:
# net = ResNet18_filtermul(filter_multiplier=2)
