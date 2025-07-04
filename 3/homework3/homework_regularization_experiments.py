import pandas as pd
from utils.experiment_utils import run_experiment, get_cifar10_data, INPUT_DIM, OUTPUT_DIM
from utils.model_utils import FullyConnectedNet
from utils.visualization_utils import plot_learning_curves, plot_weight_distribution

def regularization_experiments():
    """Проводит эксперименты с различными техниками регуляризации."""
    train_loader, test_loader = get_cifar10_data()
    epochs = 20
    lr = 1e-3
    hidden_dims = [512, 256] 
    
    reg_configs = {
        "no_reg": {'dropout_p': 0.0, 'use_batchnorm': False, 'weight_decay': 0.0},
        "dropout_0.1": {'dropout_p': 0.1, 'use_batchnorm': False, 'weight_decay': 0.0},
        "dropout_0.3": {'dropout_p': 0.3, 'use_batchnorm': False, 'weight_decay': 0.0},
        "dropout_0.5": {'dropout_p': 0.5, 'use_batchnorm': False, 'weight_decay': 0.0},
        "batchnorm": {'dropout_p': 0.0, 'use_batchnorm': True, 'weight_decay': 0.0},
        "dropout_0.3_plus_bn": {'dropout_p': 0.3, 'use_batchnorm': True, 'weight_decay': 0.0},
        "l2_reg_1e-4": {'dropout_p': 0.0, 'use_batchnorm': False, 'weight_decay': 1e-4},
    }
    
    names_ru = {
        "no_reg": "Без регуляризации", "dropout_0.1": "Dropout (p=0.1)",
        "dropout_0.3": "Dropout (p=0.3)", "dropout_0.5": "Dropout (p=0.5)",
        "batchnorm": "Только BatchNorm", "dropout_0.3_plus_bn": "Dropout + BatchNorm",
        "l2_reg_1e-4": "L2 регуляризация (1e-4)",
    }
    results = []

    print("--- Запуск экспериментов с регуляризацией (Часть 3.1 & 3.2) ---")
    for name, params in reg_configs.items():
        model = FullyConnectedNet(INPUT_DIM, hidden_dims, OUTPUT_DIM, 
                                dropout_p=params['dropout_p'], use_batchnorm=params['use_batchnorm'])
        exp_name = f"regularization_experiments/{name}"
        history_df, final_metrics = run_experiment(model, exp_name, train_loader, test_loader, 
                                                   epochs, lr, weight_decay=params['weight_decay'])
        
        plot_learning_curves(history_df, f"Регуляризация: {names_ru[name]}", f"plots/{exp_name}_curves.png")
        plot_weight_distribution(model, f"Регуляризация: {names_ru[name]}", f"plots/{exp_name}_weights.png")
        
        final_metrics['config_name'] = names_ru[name]
        results.append(final_metrics)
    
    results_df = pd.DataFrame(results).sort_values(by='final_test_acc', ascending=False)
    results_df_to_save = results_df.drop(columns=['experiment_name'])
    results_df_to_save.to_csv('results/regularization_experiments/summary.csv', index=False)
    
    print("\nИтоговые результаты по экспериментам с регуляризацией:")
    pd.set_option('display.width', 120)
    print(results_df_to_save)

if __name__ == '__main__':
    regularization_experiments()