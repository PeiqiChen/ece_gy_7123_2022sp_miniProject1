'''Train CIFAR10 with PyTorch.'''
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# set seed
setup_seed(312801)

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
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32  # 64改为32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # planes每个都减半
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)

        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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

def ResNet18():
    return ResNet(BasicBlock, [3, 6, 4, 3])

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# 指定使用0,1,2三块卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate') # 1e-3
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # AddGaussianNoise(0., 1.),
    transforms.Normalize([0.5], [0.5])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize([0.5], [0.5])
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# a = "l4-b3643-SDG-Noise01-EPOCH400"  # Adam-lr001-tmax200

# a="l4-b3643-Adam-lr001-tmax64"
# a="l4-b3643-Adam-lr01-origdataAug"
# a = "l4-b3643-SDG-lr1e-3"
a = "l4-b3643-SDG-lr01Static"

# LR = LearningRate()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./' + a + 'output/checkpoint/ckpt.pth')

    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


# optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Calculate parameters
def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
    # torch.numel() returns number of elements in a tensor


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_loss_store_temp = []
    train_acc_store_temp = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_loss_store_temp.append(train_loss / (batch_idx + 1))
        train_acc_store_temp.append(100. * correct / total)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        train_loss_temp = train_loss / (batch_idx + 1)
    # return train_loss_store_temp, train_acc_store_temp # 返回train的loss和accuracy
    return train_loss_temp, 100. * correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_store_temp = []
    test_acc_store_temp = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_loss_store_temp.append(test_loss / (batch_idx + 1))
            test_acc_store_temp.append(100. * correct / total)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            test_loss_temp = test_loss / (batch_idx + 1)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        os.makedirs('./' + a + 'output/', exist_ok=True)
        os.makedirs('./' + a + 'output/checkpoint/', exist_ok=True)
        torch.save(state, './' + a + 'output/checkpoint/project1_model.pt')
        best_acc = acc

    return test_loss_temp, 100. * correct / total  # 同train


# 储存train和test的loss和accuracy
train_loss_store = []
train_acc_store = []
test_loss_store = []
test_acc_store = []

os.makedirs('./' + a + 'output/', exist_ok=True)
os.makedirs('./' + a + 'output/checkpoint/', exist_ok=True)
f = open("./" + a + "output/" + a + "-calculate_parameters.txt", "w")
f.writelines(str(count_parameters(net)))
f.close()

print("calculate parameters: ", count_parameters(net))

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

start = time.time()
for epoch in range(start_epoch, start_epoch + 200):
    temp_a, temp_b = train(epoch)
    train_loss_store.append(temp_a)
    train_acc_store.append(temp_b)
    temp_c, temp_d = test(epoch)
    test_loss_store.append(temp_c)
    test_acc_store.append(temp_d)
    # scheduler.step()

end = time.time()

f = open("./" + a + "output/" + a + "-time_cost.txt", "w")
f.writelines(str(end - start))
f.close()
print("Time cost: ", str(end - start))

#    print("calculate parameters: ", count_parameters(net))

f = open("./" + a + "output/" + a + "-train_loss_store.txt", "w")
f.writelines(str(train_loss_store))
f.close()

f = open("./" + a + "output/" + a + "-train_acc_store.txt", "w")
f.writelines(str(train_acc_store))
f.close()

f = open("./" + a + "output/" + a + "-test_loss_store.txt", "w")
f.writelines(str(test_loss_store))
f.close()

f = open("./" + a + "output/" + a + "-test_acc_store.txt", "w")
f.writelines(str(test_acc_store))
f.close()

plt.figure(1)
plt.plot(train_loss_store, color='green', label='train_loss')
plt.plot(test_loss_store, color='skyblue', label='test_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./' + a + 'output/' + a + '-loss.jpg')
plt.show()

plt.figure(2)
plt.plot(train_acc_store, color='red', label='train_acc')
plt.plot(test_acc_store, color='blue', label='test_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('./' + a + 'output/' + a + '-accuracy.jpg')
plt.show()
