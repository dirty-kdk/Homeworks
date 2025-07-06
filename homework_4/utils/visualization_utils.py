import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_curves(history_dict, title, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    for name, history in history_dict.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'o-', label=f'{name} Train Loss')
        ax1.plot(epochs, history['val_loss'], 's--', label=f'{name} Val Loss')
        ax2.plot(epochs, history['train_acc'], 'o-', label=f'{name} Train Acc')
        ax2.plot(epochs, history['val_acc'], 's--', label=f'{name} Val Acc')
        
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(model, dataloader, device, class_names, save_path=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_kernels(model, layer_index=0, save_path=None):
    """Визуализирует ядра (фильтры) первого сверточного слоя."""
    # Находим первый сверточный слой
    first_conv_layer = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            first_conv_layer = layer
            break
            
    if first_conv_layer is None:
        print("No Conv2D layer found in the model.")
        return

    kernels = first_conv_layer.weight.data.clone().cpu()
    
    # Нормализуем для лучшей визуализации
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    
    num_kernels = kernels.shape[0]
    num_cols = 8
    num_rows = num_kernels // num_cols + (1 if num_kernels % num_cols else 0)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()
    for i in range(num_kernels):
        # Отображаем только первый входной канал, если их несколько (например, для RGB)
        kernel = kernels[i, 0, :, :]
        axes[i].imshow(kernel, cmap='gray')
        axes[i].axis('off')
    
    for j in range(num_kernels, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle('Kernels from the first convolutional layer')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_feature_maps(model, image, save_path=None):
    """Визуализирует карты признаков после нескольких слоев."""
    model.eval()
    image = image.unsqueeze(0) # Добавляем batch dimension
    outputs = []
    layer_names = []

    # Проходим по слоям и сохраняем выходы
    x = image
    for name, module in model.named_children():
        x = module(x)
        if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.ReLU, torch.nn.Sequential)):
            outputs.append(x)
            layer_names.append(str(name))

    for i, feature_map in enumerate(outputs[:5]): # Визуализируем первые 5 слоев/блоков
        feature_map = feature_map.squeeze(0).cpu().detach()
        num_features = feature_map.shape[0]
        
        num_cols = 8
        num_rows = min(4, num_features // num_cols + (1 if num_features % num_cols else 0))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        fig.suptitle(f'Feature maps after layer: {layer_names[i]}', fontsize=16)
        axes = axes.flatten()

        for j in range(num_rows * num_cols):
            if j < num_features:
                axes[j].imshow(feature_map[j], cmap='viridis')
                axes[j].axis('off')
            else:
                axes[j].axis('off')
        
        if save_path:
            plt.savefig(f"{save_path}_layer_{i}.png")
        plt.show()