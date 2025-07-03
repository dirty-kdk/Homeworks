import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from homework_datasets import CustomCSVDataset
from homework_model_modification import LinearRegression, LogisticRegression, train_model  # Добавлен импорт

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Определение модели линейной регрессии
class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

def run_experiment(dataset, model_class, in_features, num_classes=None, lr=0.01, batch_size=32, optimizer_class=optim.SGD, epochs=50):
    """Запуск эксперимента с заданными гиперпараметрами."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model_class(in_features=in_features, num_classes=num_classes) if num_classes else model_class(in_features=in_features)
    criterion = nn.CrossEntropyLoss() if num_classes else nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(1, epochs + 1):
        total_loss = 0
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            # Адаптация целевой переменной: Long для классификации, Float для регрессии
            target = batch_y.long() if num_classes else batch_y.float().unsqueeze(1)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            logger.info(f"Эпоха {epoch}, Средняя потеря: {avg_loss:.4f}")
    
    return losses

def plot_hyperparameter_results(results, filename='plots/hyperparameter_results.png'):
    """Визуализация результатов экспериментов."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.figure(figsize=(10, 6))
    for params, losses in results.items():
        lr, batch_size, optimizer = params  # Распаковка кортежа
        plt.plot(losses, label=f"lr={lr}, batch_size={batch_size}, opt={optimizer.__name__}")
    plt.xlabel('Эпоха')
    plt.ylabel('Средняя потеря')
    plt.title('Сравнение гиперпараметров')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    logger.info(f"График сохранен в {filename}")

def create_polynomial_features(X, degree=2):
    """Создание полиномиальных признаков."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

def create_interaction_features(X):
    """Создание признаков взаимодействий."""
    n_features = X.shape[1]
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append(X[:, i] * X[:, j])
    return np.column_stack(interactions)

def create_statistical_features(X):
    """Создание статистических признаков (среднее, дисперсия)."""
    mean_features = np.mean(X, axis=1, keepdims=True)
    var_features = np.var(X, axis=1, keepdims=True)
    return np.column_stack([mean_features, var_features])

def compare_models(dataset, model_class, in_features, num_classes=None):
    """Сравнение моделей с разными наборами признаков."""
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = model_class(in_features=in_features, num_classes=num_classes) if num_classes else model_class(in_features=in_features)
    criterion = nn.CrossEntropyLoss() if num_classes else nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    for epoch in range(1, 51):
        total_loss = 0
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            target = batch_y.long() if num_classes else batch_y.float().unsqueeze(1)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            logger.info(f"Эпоха {epoch}, Средняя потеря: {avg_loss:.4f}")
    
    return losses

if __name__ == '__main__':
    # Загрузка датасета
    try:
        dataset = CustomCSVDataset(
            csv_file='data/diabetes.csv',
            target_column='Outcome',
            categorical_columns=[]  # Нет категориальных столбцов
        )
        
        # Эксперименты с гиперпараметрами
        learning_rates = [0.001, 0.01, 0.1]
        batch_sizes = [16, 32, 64]
        optimizers = [optim.SGD, optim.Adam, optim.RMSprop]
        results = {}
        
        for lr in learning_rates:
            for bs in batch_sizes:
                for opt in optimizers:
                    params = {'lr': lr, 'batch_size': bs, 'optimizer': opt}
                    logger.info(f"Эксперимент: lr={lr}, batch_size={bs}, optimizer={opt.__name__}")
                    losses = run_experiment(
                        dataset=dataset,
                        model_class=LogisticRegression,
                        in_features=dataset.features.shape[1],
                        num_classes=2,
                        lr=lr,
                        batch_size=bs,
                        optimizer_class=opt,
                        epochs=50
                    )
                    results[tuple(params.values())] = losses  # Используем кортеж как ключ
        
        plot_hyperparameter_results(results)
        
        # Feature Engineering
        X, y = dataset.features.numpy(), dataset.labels.numpy()
        
        X_poly = create_polynomial_features(X, degree=2)
        X_inter = create_interaction_features(X)
        X_stat = create_statistical_features(X)
        
        X_enhanced = np.column_stack([X, X_poly, X_inter, X_stat])
        enhanced_dataset = TensorDataset(
            torch.tensor(X_enhanced, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
        
        baseline_losses = compare_models(dataset, LogisticRegression, in_features=X.shape[1], num_classes=2)
        enhanced_losses = compare_models(enhanced_dataset, LogisticRegression, in_features=X_enhanced.shape[1], num_classes=2)
        
        plt.figure(figsize=(10, 6))
        plt.plot(baseline_losses, label='Базовая модель')
        plt.plot(enhanced_losses, label='Модель с новыми признаками')
        plt.xlabel('Эпоха')
        plt.ylabel('Средняя потеря')
        plt.title('Сравнение моделей с разными признаками')
        plt.legend()
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/feature_engineering_comparison_diabetes.png')
        plt.close()
        logger.info("График сравнения сохранен в plots/feature_engineering_comparison_diabetes.png")
    except FileNotFoundError:
        logger.error("Файл csv не найден")