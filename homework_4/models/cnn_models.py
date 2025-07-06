import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 28x28 -> pool -> 14x14 -> pool -> 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResCNN_MNIST(nn.Module):
    def __init__(self, block=ResidualBlock, num_classes=10):
        super(ResCNN_MNIST, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, 2, stride=1)
        self.layer2 = self._make_layer(block, 32, 2, stride=2) # 28 -> 14
        self.layer3 = self._make_layer(block, 64, 2, stride=2) # 14 -> 7
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Для Задания 1.2, 2.1, 2.2
class GenericCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, kernel_size=3, depth_config=None, use_residual=False, use_regularization=False):
        super(GenericCNN, self).__init__()
        self.use_regularization = use_regularization
        
        if depth_config is None:
             depth_config = [(32, 2), (64, 2), (128, 2)] # (channels, num_blocks)

        layers = [nn.Conv2d(in_channels, depth_config[0][0], kernel_size=kernel_size, padding=kernel_size//2)]
        
        current_channels = depth_config[0][0]
        
        for channels, num_blocks in depth_config:
            for i in range(num_blocks):
                block = []
                if use_residual:
                    block.append(ResidualBlock(current_channels, channels))
                else:
                    block.append(nn.Conv2d(current_channels, channels, kernel_size=kernel_size, padding=kernel_size//2))
                    block.append(nn.BatchNorm2d(channels))
                    block.append(nn.ReLU(inplace=True))
                
                if self.use_regularization and i % 2 == 1:
                    block.append(nn.Dropout(0.2))

                layers.append(nn.Sequential(*block))
                current_channels = channels
            layers.append(nn.MaxPool2d(2)) # Downsample after each major block

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # This part is tricky as final size depends on depth. Let's use adaptive pooling to fix this.
        self.classifier = nn.Linear(depth_config[-1][0], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CombinedKernelCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1x1 = nn.Conv2d(3, 16, kernel_size=1, padding=0)
        self.conv1_3x3 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # 16 + 16 channels
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.classifier = nn.Linear(64 * 8 * 8, num_classes) # CIFAR 32x32 -> 16x16 -> 8x8
        
    def forward(self, x):
        out_1x1 = self.conv1_1x1(x)
        out_3x3 = self.conv1_3x3(x)
        
        # Concatenate along the channel dimension
        out = torch.cat([out_1x1, out_3x3], dim=1)
        out = self.pool(self.relu(self.bn1(out)))
        
        out = self.pool(self.relu(self.bn2(self.conv2(out))))
        
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out