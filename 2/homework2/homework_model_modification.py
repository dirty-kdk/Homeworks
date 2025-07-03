# Задание 1: Модификация существующих моделей
# Создайте файл homework_model_modification.py:

# 1.1 Расширение линейной регрессии
# # Модифицируйте существующую линейную регрессию:
# # - Добавьте L1 и L2 регуляризацию
# # - Добавьте early stopping
# 1.2 Расширение логистической регрессии
# # Модифицируйте существующую логистическую регрессию:
# # - Добавьте поддержку многоклассовой классификации
# # - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
# # - Добавьте визуализацию confusion matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Предполагается, что утилиты доступны
from utils import make_regression_data, mse, log_epoch, RegressionDataset

class LinearRegression(nn.Module):
    def __init__(self, in_features, l1_lambda=0.01, l2_lambda=0.01):
        """
        Инициализация модели линейной регрессии с L1 и L2 регуляризацией.
        
        Аргументы:
            in_features (int): Количество входных признаков.
            l1_lambda (float): Коэффициент L1 регуляризации.
            l2_lambda (float): Коэффициент L2 регуляризации.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        """Вычисление L1 и L2 регуляризации."""
        l1_loss = torch.tensor(0., requires_grad=True)
        l2_loss = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l1_loss = l1_loss + torch.norm(param, 1)
            l2_loss = l2_loss + torch.norm(param, 2) ** 2
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        """
        Инициализация модели логистической регрессии для многоклассовой классификации.
        
        Аргументы:
            in_features (int): Количество входных признаков.
            num_classes (int): Количество классов.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))

def train_model(model, dataloader, criterion, optimizer, epochs, patience=10):
    """
    Обучение модели с early stopping.
    
    Аргументы:
        model: Модель для обучения.
        dataloader: DataLoader с данными.
        criterion: Функция потерь.
        optimizer: Оптимизатор.
        epochs (int): Количество эпох.
        patience (int): Количество эпох для early stopping.
    """
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        total_loss = 0
        model.train()
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            if isinstance(model, LinearRegression):
                loss += model.regularization_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (i + 1)
        logger.info(f'Эпоха {epoch}, Средняя потеря: {avg_loss:.4f}')
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Ранняя остановка на эпохе {epoch}")
                model.load_state_dict(best_model_state)
                break
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

def compute_metrics(y_true, y_pred, y_prob, num_classes):
    """
    Вычисление метрик: precision, recall, F1-score, ROC-AUC.
    
    Аргументы:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.
        y_prob: Вероятности классов.
        num_classes: Количество классов.
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    if num_classes == 2:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    return precision, recall, f1, roc_auc

def plot_confusion_matrix(y_true, y_pred, num_classes, filename='plots/confusion_matrix.png'):
    """Визуализация матрицы ошибок."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.savefig(filename)
    plt.close()
    logger.info(f"Матрица ошибок сохранена в {filename}")

if __name__ == '__main__':
    # Линейная регрессия
    X, y = make_regression_data(n=200)
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    logger.info(f'Размер датасета: {len(dataset)}, Количество батчей: {len(dataloader)}')

    model = LinearRegression(in_features=1, l1_lambda=0.01, l2_lambda=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    train_model(model, dataloader, criterion, optimizer, epochs=100, patience=10)

    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/linreg_torch.pth')
    logger.info("Модель линейной регрессии сохранена в models/linreg_torch.pth")

    # Логистическая регрессия
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=4, n_classes=3, n_clusters_per_class=1, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LogisticRegression(in_features=4, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    train_model(model, dataloader, criterion, optimizer, epochs=100, patience=10)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_prob = y_pred.numpy()
        y_pred = torch.argmax(y_pred, dim=1).numpy()
        y_true = y.numpy()
    
    compute_metrics(y_true, y_pred, y_prob, num_classes=3)
    plot_confusion_matrix(y_true, y_pred, num_classes=3)
    
    torch.save(model.state_dict(), 'models/logreg_torch.pth')
    logger.info("Модель логистической регрессии сохранена в models/logreg_torch.pth")