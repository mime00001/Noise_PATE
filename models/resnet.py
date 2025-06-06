import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#this code is taken from https://github.com/Piyush-555/GaussianDistillation/tree/main

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
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

def ResNet10(*args):
    return ResNet(BasicBlock, [1, 1, 1, 1])

def ResNet12(*args):
    return ResNet(BasicBlock, [2, 1, 1, 1])

def ResNet18(*args):
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18_MNIST(*args):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                        stride=1, padding=1, bias=False)
    return model
    
def ResNet34(*args):
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(*args):
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101(*args):
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152(*args):
    return ResNet(Bottleneck, [3, 8, 36, 3])


def MNISTResNet(*args):
    self = ResNet(BasicBlock, [1, 1, 1, 1])
    self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                        stride=1, padding=1, bias=False)
    return self
    




def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def train_step(self, batch, model, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, labels)
        return loss

    def validation_step(self, batch, model, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        acc = accuracy(output, labels)
        return acc


def conv_bn_relu_pool(in_channels, out_channels, pool=False):
    net = nn.Sequential(
        OrderedDict([
            ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)),
            ('batch', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))
    if pool:
        net = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)),
                ('batch', nn.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU(inplace=True)),
                ('pool', nn.MaxPool2d(2))
            ]))
    return net


class ResNet9class(Base):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.prep = conv_bn_relu_pool(in_channels, 32, pool=True)
        self.layer1_head = conv_bn_relu_pool(32, 64, pool=True)
        self.layer1_residual1 = conv_bn_relu_pool(64, 64)
        self.layer1_residual2 = conv_bn_relu_pool(64, 64)
        self.layer2 = conv_bn_relu_pool(64, 128, pool=True)
        self.layer3_head = conv_bn_relu_pool(128, 256, pool=True)
        self.layer3_residual1 = conv_bn_relu_pool(256, 256)
        self.layer3_residual2 = conv_bn_relu_pool(256, 256)
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.prep(x)

        x = self.layer1_head(x)

        a = x
        x = self.layer1_residual1(x)

        x = self.layer1_residual2(x) + a

        x = self.layer2(x)

        x = self.layer3_head(x)

        b = x
        x = self.layer3_residual1(x)

        x = self.layer3_residual2(x) + b

        x = self.classifier(x)

        return x
    
def ResNet9(*args):
    return ResNet9class()