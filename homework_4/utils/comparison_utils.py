import torch
import time

def count_parameters(model):
    """Считает количество обучаемых параметров в модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, dummy_input, device, repetitions=100):
    """Измеряет среднее время инференса модели."""
    model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()
    
    # "Прогрев" CUDA
    for _ in range(10):
        _ = model(dummy_input)
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / repetitions * 1000
    return avg_time_ms