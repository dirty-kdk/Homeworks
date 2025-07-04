import pandas as pd
from utils.experiment_utils import run_experiment, get_cifar10_data, INPUT_DIM, OUTPUT_DIM
from utils.model_utils import FullyConnectedNet, count_parameters
from utils.visualization_utils import plot_learning_curves, plot_comparison, plot_heatmap

def width_experiments():
    """Проводит эксперименты с моделями разной ширины."""
    train_loader, test_loader = get_cifar10_data()
    epochs = 15
    lr = 1e-3
    
    width_configs = {
        "uzkaia": [64, 32],
        "sredniaia": [256, 128],
        "shirokaia": [1024, 512],
        "ochen_shirokaia": [2048, 1024],
    }
    
    names_ru = {"uzkaia": "Узкая", "sredniaia": "Средняя", "shirokaia": "Широкая", "ochen_shirokaia": "Очень широкая"}
    results = []

    print("--- Запуск экспериментов с шириной сети (Часть 2.1) ---")
    for name, hidden_dims in width_configs.items():
        model = FullyConnectedNet(INPUT_DIM, hidden_dims, OUTPUT_DIM)
        exp_name = f"width_experiments/{name}"
        
        history_df, final_metrics = run_experiment(model, exp_name, train_loader, test_loader, epochs, lr)
        plot_learning_curves(history_df, f"Ширина: {names_ru[name]}", f"plots/{exp_name}.png")
        
        final_metrics['width_name'] = name
        final_metrics['params'] = count_parameters(model)
        results.append(final_metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/width_experiments/summary.csv', index=False)
    
    plot_df = results_df.rename(columns={'final_test_acc': 'Итоговая точность', 'training_time_sec': 'Время обучения (сек)'})
    plot_comparison(plot_df, 'params', ['Итоговая точность', 'Время обучения (сек)'], 
                    'Влияние ширины (параметров) на точность и время', 'plots/width_experiments/summary_comparison.png')

def grid_search_architecture():
    """Выполняет поиск по сетке для нахождения оптимальной архитектуры."""
    train_loader, test_loader = get_cifar10_data()
    epochs = 15
    lr = 1e-3

    base_widths = [128, 256, 512]
    schemes = ['constant', 'contracting', 'expanding']
    schemes_ru = {'constant': 'Постоянная', 'contracting': 'Сужающаяся', 'expanding': 'Расширяющаяся'}
    results_grid = pd.DataFrame(index=[schemes_ru[s] for s in schemes], columns=[str(w) for w in base_widths], dtype=float)

    print("\n--- Запуск поиска по сетке архитектур (Часть 2.2) ---")
    for scheme in schemes:
        for width in base_widths:
            hidden_dims = [width, width] if scheme == 'constant' else [width, width // 2] if scheme == 'contracting' else [width, width * 2]
            model = FullyConnectedNet(INPUT_DIM, hidden_dims, OUTPUT_DIM)
            exp_name = f"width_experiments/grid_search_{scheme}_{width}"
            
            _, final_metrics = run_experiment(model, exp_name, train_loader, test_loader, epochs, lr)
            results_grid.loc[schemes_ru[scheme], str(width)] = final_metrics['final_test_acc']
            
    results_grid.to_csv('results/width_experiments/grid_search_results.csv')
    plot_heatmap(results_grid, 'Поиск оптимальной архитектуры по сетке (Точность на тесте)', 
                 'plots/width_experiments/grid_search_heatmap.png',
                 'Базовая ширина', 'Схема')

if __name__ == '__main__':
    width_experiments()
    grid_search_architecture()