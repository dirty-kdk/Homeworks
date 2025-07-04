import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_learning_curves(history, title, save_path):
    """Строит и сохраняет графики кривых обучения (потери и точность)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Потери (Train)')
    plt.plot(history['epoch'], history['test_loss'], label='Потери (Test)')
    plt.title(f'{title} - Потери')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Точность (Train)')
    plt.plot(history['epoch'], history['test_acc'], label='Точность (Test)')
    plt.title(f'{title} - Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison(results_df, x_col, y_cols, title, save_path):
    """Строит график для сравнения нескольких метрик."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for y_col in y_cols:
        plt.plot(results_df[x_col], results_df[y_col], marker='o', label=y_col)
    
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(data, title, save_path, xlabel, ylabel):
    """Строит и сохраняет тепловую карту."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(data.astype(float), annot=True, fmt=".4f", cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()

def plot_weight_distribution(model, title, save_path):
    """Строит гистограмму распределения весов модели."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            weights.extend(param.data.cpu().numpy().flatten())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(weights, bins=100, kde=True)
    plt.title(f'Распределение весов - {title}')
    plt.xlabel('Значение веса')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()