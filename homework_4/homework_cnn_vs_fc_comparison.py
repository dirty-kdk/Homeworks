import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import os

from models.fc_models import FC_MNIST, FC_CIFAR_Deep
from models.cnn_models import SimpleCNN_MNIST, ResCNN_MNIST, GenericCNN
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_curves, plot_confusion_matrix
from utils.comparison_utils import count_parameters, measure_inference_time

# --- Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs('plots', exist_ok=True)
os.makedirs('results/mnist_comparison', exist_ok=True)
os.makedirs('results/cifar_comparison', exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 5 # Увеличьте до 10-15 для лучших результатов
LR = 0.001

# --- 1.1 Сравнение на MNIST ---
def run_mnist_comparison():
    logging.info("Starting MNIST Comparison (Task 1.1)")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    models_mnist = {
        "FC (3-layers)": FC_MNIST(),
        "Simple CNN (2-conv)": SimpleCNN_MNIST(),
        "CNN with Residuals": ResCNN_MNIST()
    }
    
    histories = {}
    results = {}
    for name, model in models_mnist.items():
        model.to(DEVICE)
        logging.info(f"Model: {name}")
        params = count_parameters(model)
        logging.info(f"Parameters: {params:,}")
        
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(model, train_loader, test_loader, criterion, optimizer, DEVICE, num_epochs=EPOCHS, model_name=name)
        histories[name] = history
        
        inference_time = measure_inference_time(model, torch.randn(1, 1, 28, 28), DEVICE)
        final_acc = history['val_acc'][-1]
        logging.info(f"Final Validation Accuracy: {final_acc:.4f}")
        logging.info(f"Average Inference Time: {inference_time:.4f} ms/sample")
        results[name] = {'params': params, 'accuracy': final_acc, 'inference_time': inference_time}
    
    plot_training_curves(histories, "MNIST Model Comparison", "plots/mnist_comparison_curves.png")
    
    logging.info("--- MNIST Comparison Summary ---")
    for name, res in results.items():
        logging.info(f"{name:<20} | Params: {res['params']:,} | Accuracy: {res['accuracy']:.4f} | Inference Time (ms): {res['inference_time']:.4f}")


# --- 1.2 Сравнение на CIFAR-10 ---
def run_cifar_comparison():
    logging.info("Starting CIFAR-10 Comparison (Task 1.2)")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    models_cifar = {
        "Deep FC": FC_CIFAR_Deep(),
        "CNN with Residuals": GenericCNN(use_residual=True),
        "CNN with Regularization + Residuals": GenericCNN(use_residual=True, use_regularization=True)
    }

    histories = {}
    results = {}
    for name, model in models_cifar.items():
        model.to(DEVICE)
        logging.info(f"Model: {name}")
        params = count_parameters(model)
        logging.info(f"Parameters: {params:,}")

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        
        history = train_model(model, train_loader, test_loader, criterion, optimizer, DEVICE, num_epochs=EPOCHS+5, model_name=name) # CIFAR requires more epochs
        histories[name] = history
        
        inference_time = measure_inference_time(model, torch.randn(1, 3, 32, 32), DEVICE)
        final_acc = history['val_acc'][-1]
        logging.info(f"Final Validation Accuracy: {final_acc:.4f}")
        logging.info(f"Average Inference Time: {inference_time:.4f} ms/sample")
        results[name] = {'params': params, 'accuracy': final_acc, 'inference_time': inference_time}

        class_names = train_dataset.classes
        plot_confusion_matrix(model, test_loader, DEVICE, class_names, f"plots/cifar_cm_{name.replace(' ', '_')}.png")
        
    plot_training_curves(histories, "CIFAR-10 Model Comparison", "plots/cifar_comparison_curves.png")
    
    logging.info("--- CIFAR-10 Comparison Summary ---")
    for name, res in results.items():
        logging.info(f"{name:<40} | Params: {res['params']:,} | Accuracy: {res['accuracy']:.4f} | Inference Time (ms): {res['inference_time']:.4f}")

if __name__ == "__main__":
    run_mnist_comparison()
    print("\n" + "="*50 + "\n")
    run_cifar_comparison()