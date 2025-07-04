import torch.nn as nn

class FullyConnectedNet(nn.Module):
    """
    Полносвязная нейронная сеть.

    Аргументы:
        input_dim (int): Размерность входных данных.
        hidden_dims (list): Список с размерностями скрытых слоев.
        output_dim (int): Размерность выходного слоя.
        dropout_p (float): Вероятность dropout.
        use_batchnorm (bool): Использовать ли BatchNorm.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0, use_batchnorm=False):
        super(FullyConnectedNet, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout = nn.Dropout(p=dropout_p)

        self.layers = nn.ModuleList()
        current_dim = input_dim

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(self.dropout)
            current_dim = h_dim
        
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

def count_parameters(model):
    """Подсчитывает количество обучаемых параметров в модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)