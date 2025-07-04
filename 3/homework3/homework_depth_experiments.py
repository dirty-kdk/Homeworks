import pandas as pd
from utils.experiment_utils import run_experiment, get_cifar10_data, INPUT_DIM, OUTPUT_DIM
from utils.model_utils import FullyConnectedNet, count_parameters
from utils.visualization_utils import plot_learning_curves, plot_comparison

def depth_experiments():
    """Проводит эксперименты с моделями разной глубины."""
    train_loader, test_loader = get_cifar10_data()
    epochs = 20
    lr = 1e-3
    
    depth_configs = {
        "1_sloy": [],
        "2_sloia": [512],
        "3_sloia": [512, 256],
        "5_sloev": [512, 256, 128, 64],
        "7_sloev": [512, 512, 256, 256, 128, 64],
    }

    results = []
    
    print("--- Запуск экспериментов с глубиной сети (Часть 1.1) ---")
    for name, hidden_dims in depth_configs.items():
        model = FullyConnectedNet(INPUT_DIM, hidden_dims, OUTPUT_DIM)
        exp_name = f"depth_experiments/{name}"
        history_df, final_metrics = run_experiment(model, exp_name, train_loader, test_loader, epochs, lr)
        
        plot_learning_curves(history_df, f"Глубина: {name}", f"plots/{exp_name}.png")
        
        final_metrics['num_layers'] = len(hidden_dims) + 1
        final_metrics['params'] = count_parameters(model)
        results.append(final_metrics)

    results_df = pd.DataFrame(results)
    plot_df = results_df.rename(columns={
        'final_test_acc': 'Итоговая точность', 
        'training_time_sec': 'Время обучения (сек)'
    })
    
    plot_comparison(plot_df, 'num_layers', ['Итоговая точность', 'Время обучения (сек)'], 
                    'Влияние глубины на точность и время', 'plots/depth_experiments/summary_comparison.png')

def overfitting_experiments():
    """Исследует переобучение на глубокой модели и эффект регуляризации."""
    train_loader, test_loader = get_cifar10_data()
    epochs = 30 
    lr = 1e-3
    hidden_dims_deep = [512, 512, 256, 256, 128, 64]

    configs = {
        "deep_no_reg": {'dropout_p': 0.0, 'use_batchnorm': False},
        "deep_with_dropout": {'dropout_p': 0.5, 'use_batchnorm': False},
        "deep_with_bn": {'dropout_p': 0.0, 'use_batchnorm': True},
        "deep_with_dropout_bn": {'dropout_p': 0.5, 'use_batchnorm': True},
    }
    
    names_ru = {
        "deep_no_reg": "Без регуляризации",
        "deep_with_dropout": "С Dropout",
        "deep_with_bn": "С BatchNorm",
        "deep_with_dropout_bn": "Dropout + BN"
    }

    print("\n--- Запуск анализа переобучения (Часть 1.2) ---")
    for name, params in configs.items():
        model = FullyConnectedNet(INPUT_DIM, hidden_dims_deep, OUTPUT_DIM, **params)
        exp_name = f"depth_experiments/{name}"
        history_df, _ = run_experiment(model, exp_name, train_loader, test_loader, epochs, lr)
        plot_learning_curves(history_df, f"Анализ переобучения: {names_ru[name]}", f"plots/{exp_name}.png")

if __name__ == '__main__':
    depth_experiments()
    overfitting_experiments()