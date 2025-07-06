import torch
import torch.nn as nn
import logging

from models.custom_layers import (
    CustomConvWithScaling, SimpleSEAttention, CustomLeakyGELU, LpPooling,
    BasicResidualBlock, BottleneckResidualBlock, WideResidualBlock
)
from utils.comparison_utils import count_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_custom_layers_demo():
    logging.info("--- 3.1 Custom Layers Demonstration ---")
    dummy_input = torch.randn(2, 16, 32, 32) # (B, C, H, W)

    # 1. Кастомный сверточный слой
    logging.info("\n--- CustomConvWithScaling ---")
    custom_conv = CustomConvWithScaling(16, 32, kernel_size=3, padding=1)
    standard_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    logging.info(f"Custom Conv Params: {count_parameters(custom_conv)}")
    logging.info(f"Standard Conv Params: {count_parameters(standard_conv)}") # Должно быть на 32 параметра меньше (без scale)
    output = custom_conv(dummy_input)
    logging.info(f"Output shape: {output.shape}")
    
    # 2. Attention механизм
    logging.info("\n--- SimpleSEAttention ---")
    attention = SimpleSEAttention(channel=16, reduction=4)
    output = attention(dummy_input)
    logging.info(f"Attention output shape (should be same as input): {output.shape}")
    
    # 3. Кастомная функция активации
    logging.info("\n--- CustomLeakyGELU ---")
    activation = CustomLeakyGELU()
    test_tensor = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    output = activation(test_tensor)
    logging.info(f"Input to activation: {test_tensor}")
    logging.info(f"Output from CustomLeakyGELU: {output}")

    # 4. Кастомный pooling слой
    logging.info("\n--- LpPooling ---")
    lp_pool = LpPooling(p_norm=2, kernel_size=2, stride=2) # L2-pooling
    output = lp_pool(dummy_input)
    logging.info(f"LpPooling output shape: {output.shape}")

def run_residual_block_experiments():
    logging.info("\n--- 3.2 Residual Block Variants Experiment ---")
    in_channels = 64
    out_channels = 64
    dummy_input = torch.randn(2, in_channels, 32, 32)
    
    blocks = {
        "Basic Block": BasicResidualBlock(in_channels, out_channels),
        "Bottleneck Block": BottleneckResidualBlock(in_channels, out_channels // BottleneckResidualBlock.expansion),
        "Wide Block (k=2)": WideResidualBlock(in_channels, out_channels, widen_factor=2)
    }

    logging.info("--- Comparing Residual Block Variants (Parameters & Output Shape) ---")
    for name, block in blocks.items():
        params = count_parameters(block)
        output = block(dummy_input)
        logging.info(f"{name:<20} | Params: {params:,} | Output Shape: {output.shape}")

    # Анализ:
    # Basic Block - стандартный.
    # Bottleneck Block - использует 1x1 свертки для сжатия и расширения каналов.
    #   Это значительно уменьшает количество параметров по сравнению с Basic, особенно в глубоких сетях.
    # Wide Block - увеличивает ширину (количество каналов) внутри блока, а не глубину (количество слоев).
    #   Это делает сеть "шире" и часто дает лучшие результаты, но за счет большего числа параметров.

    # Дальнейший шаг (не реализован, но требуется по заданию) - встроить эти блоки
    # в полноценную сеть (как GenericCNN), обучить на CIFAR-10 и сравнить
    # производительность и стабильность обучения (кривые loss/accuracy).
    logging.info("\nFull training comparison would be the next step to evaluate performance and stability.")

if __name__ == '__main__':
    run_custom_layers_demo()
    run_residual_block_experiments()