# Задание 2: Работа с датасетами
# Создайте файл homework_datasets.py:

# 2.1 Кастомный Dataset класс
# # Создайте кастомный класс датасета для работы с CSV файлами:
# # - Загрузка данных из файла
# # - Предобработка (нормализация, кодирование категорий)
# # - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)
# 2.2 Эксперименты с различными датасетами
# # Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, target_column, categorical_columns=None, transform=None):
        """
        Кастомный датасет для работы с CSV-файлами.
        
        Аргументы:
            csv_file (str): Путь к CSV-файлу.
            target_column (str): Название целевого столбца.
            categorical_columns (list): Список категориальных столбцов.
            transform (callable, optional): Преобразование данных.
        """
        self.data = pd.read_csv(csv_file)
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.transform = transform
        
        # Предобработка
        self._preprocess()
        
    def _preprocess(self):
        """Предобработка данных: нормализация и кодирование."""
        # Кодирование категориальных столбцов (в данном случае не требуется, так как все числовые)
        self.label_encoders = {}
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col])
        
        # Нормализация числовых столбцов
        self.scaler = StandardScaler()
        numeric_columns = [col for col in self.data.columns if col not in self.categorical_columns + [self.target_column]]
        if numeric_columns:
            self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])
        
        # Разделение на признаки и целевую переменную
        self.X = self.data.drop(columns=[self.target_column]).values
        self.y = self.data[self.target_column].values
        
        # Преобразование в тензоры
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)  # Outcome как целочисленные метки (0 или 1)
        
        logger.info(f"Датасет загружен: {len(self.X)} строк, {self.X.shape[1]} признаков")

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def experiment_with_datasets():
    """Эксперименты с датасетом для бинарной классификации."""
    from homework_model_modification import LogisticRegression, train_model
    import torch.nn as nn
    import torch.optim as optim
    
    # Проверка наличия папки models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Датасет для классификации
    try:
        diabetes_dataset = CustomCSVDataset(
            csv_file='data/diabetes.csv',
            target_column='Outcome',
            categorical_columns=[]
        )
        diabetes_loader = DataLoader(diabetes_dataset, batch_size=32, shuffle=True)
        
        diabetes_model = LogisticRegression(in_features=diabetes_dataset.X.shape[1], num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(diabetes_model.parameters(), lr=0.01)
        logger.info("Обучение логистической регрессии на датасете")
        train_model(diabetes_model, diabetes_loader, criterion, optimizer, epochs=100)
        torch.save(diabetes_model.state_dict(), 'models/logreg_diabetes.pth')
        logger.info("Модель сохранена в models/logreg_diabetes.pth")
    except FileNotFoundError:
        logger.error("Файл csv не найден")

if __name__ == '__main__':
    experiment_with_datasets()