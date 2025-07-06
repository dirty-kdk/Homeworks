import torch
import torch.nn as nn
import torch.nn.functional as F

# 3.1 Реализация кастомных слоев
class CustomConvWithScaling(nn.Module):
    """Кастомный сверточный слой с дополнительной обучаемой шкалой на выходе."""
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        # Дополнительный параметр - скаляр для каждого канала
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
    
    def forward(self, x):
        x = self.conv(x)
        return x * self.scale

class SimpleSEAttention(nn.Module):
    """Простой Squeeze-and-Excitation Attention механизм."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CustomLeakyGELU(nn.Module):
    """Кастомная функция активации: комбинация LeakyReLU и GELU."""
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.gelu = nn.GELU()

    def forward(self, x):
        # Применяем GELU для положительных значений и LeakyReLU для отрицательных
        return torch.where(x > 0, self.gelu(x), x * self.negative_slope)

class LpPooling(nn.Module):
    """Кастомный pooling слой (L-p pooling)."""
    def __init__(self, p_norm, kernel_size, stride):
        super().__init__()
        self.p_norm = p_norm
        self.pool = nn.AvgPool2d(kernel_size, stride, ceil_mode=True)
    
    def forward(self, x):
        # Lp-pooling: (mean(x^p))^(1/p)
        return self.pool(x.pow(self.p_norm)).pow(1./self.p_norm)


# 3.2 Варианты Residual блоков
class BasicResidualBlock(nn.Module):
    # Это тот же ResidualBlock из cnn_models.py, но мы дублируем для ясности
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
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
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class BottleneckResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        bottleneck_channels = out_channels
        out_channels = out_channels * self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)

class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2):
        super().__init__()
        widened_channels = out_channels * widen_factor
        
        self.conv1 = nn.Conv2d(in_channels, widened_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widened_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(widened_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)