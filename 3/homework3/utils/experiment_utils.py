import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pandas as pd
import logging
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 32 * 32 * 3
OUTPUT_DIM = 10

def get_cifar10_data(batch_size=128):
    """Загружает и подготавливает датасет CIFAR-10."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Выполняет одну эпоху обучения модели."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Обучение", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    
    return running_loss / total_samples, correct_predictions / total_samples

def evaluate(model, dataloader, criterion, device):
    """Выполняет оценку модели на данных."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Оценка", leave=False)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    return running_loss / total_samples, correct_predictions / total_samples

def run_experiment(model, experiment_name, train_loader, test_loader, epochs, lr, weight_decay=0.0):
    """Запускает полный цикл эксперимента: обучение и оценку модели."""
    os.makedirs(os.path.dirname(f'results/{experiment_name}.csv'), exist_ok=True)
    log_path = f'results/{experiment_name}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()])

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    logging.info(f"Запуск эксперимента: {experiment_name}")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        logging.info(f'Эпоха {epoch+1}/{epochs} | Потери(Train): {train_loss:.4f}, Точность(Train): {train_acc:.4f} | '
                     f'Потери(Test): {test_loss:.4f}, Точность(Test): {test_acc:.4f}')

    total_time = time.time() - start_time
    logging.info(f"Эксперимент {experiment_name} завершен за {total_time:.2f} секунд.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(f'results/{experiment_name}.csv', index=False)
    
    final_metrics = {
        'experiment_name': experiment_name,
        'final_test_acc': history['test_acc'][-1],
        'training_time_sec': total_time
    }
    return history_df, final_metrics