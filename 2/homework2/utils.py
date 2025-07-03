import torch
import numpy as np
from sklearn.datasets import make_regression

def make_regression_data(n=200, n_features=1, noise=0.1, random_state=42):
    """
    Генерация данных для регрессии.
    
    Аргументы:
        n (int): Количество примеров.
        n_features (int): Количество признаков.
        noise (float): Уровень шума.
        random_state (int): Сид для воспроизводимости.
    
    Возвращает:
        X (torch.Tensor): Признаки.
        y (torch.Tensor): Целевая переменная.
    """
    X, y = make_regression(n_samples=n, n_features=n_features, noise=noise, random_state=random_state)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    return X, y

def mse(y_pred, y_true):
    """
    Вычисление среднеквадратичной ошибки (MSE).
    
    Аргументы:
        y_pred (torch.Tensor): Предсказанные значения.
        y_true (torch.Tensor): Истинные значения.
    
    Возвращает:
        float: Значение MSE.
    """
    return ((y_pred - y_true) ** 2).mean().item()

def log_epoch(epoch, loss):
    """
    Логирование информации об эпохе.
    
    Аргументы:
        epoch (int): Номер эпохи.
        loss (float): Значение функции потерь.
    """
    print(f"Эпоха {epoch}, Потеря: {loss:.4f}")

class RegressionDataset(torch.utils.data.Dataset):
    """
    Кастомный датасет для регрессии.
    
    Аргументы:
        X (torch.Tensor): Признаки.
        y (torch.Tensor): Целевая переменная.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]