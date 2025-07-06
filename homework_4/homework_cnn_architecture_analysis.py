import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import os

from models.cnn_models import GenericCNN, CombinedKernelCNN
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_curves, visualize_kernels, visualize_feature_maps
from utils.comparison_utils import count_parameters

# --- Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs('plots/architecture', exist_ok=True)
os.makedirs('results/architecture_analysis', exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 10 # More epochs for architecture analysis
LR = 0.001

# --- Загрузка данных CIFAR-10 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- 2.1 Влияние размера ядра свертки ---
def run_kernel_size_analysis():
    logging.info("Starting Kernel Size Analysis (Task 2.1)")
    
    models_kernel = {
        "Kernel 3x3": GenericCNN(kernel_size=3),
        "Kernel 5x5": GenericCNN(kernel_size=5),
        "Kernel 7x7": GenericCNN(kernel_size=7),
        "Combined 1x1 + 3x3": CombinedKernelCNN() # Special model
    }
    
    histories = {}
    for name, model in models_kernel.items():
        model.to(DEVICE)
        logging.info(f"Model: {name}")
        params = count_parameters(model)
        # Note: maintaining same params is hard with different kernels. We just observe the effect.
        logging.info(f"Parameters: {params:,}")

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(model, train_loader, test_loader, criterion, optimizer, DEVICE, num_epochs=EPOCHS, model_name=name)
        histories[name] = history

        # Визуализация ядер первого слоя
        visualize_kernels(model, save_path=f"plots/architecture/kernels_{name.replace(' ', '_')}.png")
        
    plot_training_curves(histories, "Kernel Size Influence on CIFAR-10", "plots/architecture/kernel_size_curves.png")

# --- 2.2 Влияние глубины CNN ---
def run_depth_analysis():
    logging.info("Starting CNN Depth Analysis (Task 2.2)")
    
    # shallow: 2 conv layers
    shallow_config = [(32, 2)] 
    # medium: 4 conv layers
    medium_config = [(32, 2), (64, 2)]
    # deep: 6+ conv layers
    deep_config = [(32, 2), (64, 2), (128, 2)]

    models_depth = {
        "Shallow CNN (2 conv)": GenericCNN(depth_config=shallow_config, use_residual=False),
        "Medium CNN (4 conv)": GenericCNN(depth_config=medium_config, use_residual=False),
        "Deep CNN (6 conv)": GenericCNN(depth_config=deep_config, use_residual=False),
        "Deep CNN with Residuals": GenericCNN(depth_config=deep_config, use_residual=True)
    }

    histories = {}
    for name, model in models_depth.items():
        model.to(DEVICE)
        logging.info(f"Model: {name}")
        params = count_parameters(model)
        logging.info(f"Parameters: {params:,}")

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(model, train_loader, test_loader, criterion, optimizer, DEVICE, num_epochs=EPOCHS, model_name=name)
        histories[name] = history

        # Анализ vanishing/exploding gradients можно косвенно оценить по кривым обучения
        # и стабильности. Прямой анализ требует хуков на градиенты, что усложняет код.
        # Если модель не учится (loss/acc не меняется), это признак проблемы с градиентами.

    # Визуализация карт признаков для одной модели (например, глубокой)
    sample_img, _ = test_dataset[0]
    sample_img = sample_img.to(DEVICE)
    deep_res_model = models_depth["Deep CNN with Residuals"]
    visualize_feature_maps(deep_res_model.features, sample_img, "plots/architecture/feature_maps_deep_res")

    plot_training_curves(histories, "CNN Depth Influence on CIFAR-10", "plots/architecture/depth_influence_curves.png")


if __name__ == '__main__':
    run_kernel_size_analysis()
    print("\n" + "="*50 + "\n")
    run_depth_analysis()