import os
import csv
import time
import torch

class ExperimentTracker:
    """
    Módulo encargado de registrar las métricas de rendimiento de cada experimento,
    calcular el coste computacional y persistir los resultados en formato CSV.
    """
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.output_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.output_dir, "metrics.csv")
        self.history = []
        
        # Campos que se guardarán en el CSV
        self.headers = [
            "epoch", "train_loss", "train_acc", 
            "test_loss", "test_acc", 
            "epoch_time_sec", "inference_time_ms_per_sample",
            "max_vram_mb", "model_flops", "model_params"
        ]
        
        # Inicializar el archivo CSV con sus cabeceras
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def compute_model_stats(self, model: torch.nn.Module, input_size=(1, 3, 224, 224)):
        """
        Calcula el número de parámetros totales y estima los FLOPs del modelo.
        """
        # Contar parámetros totales y aprendibles
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimación aproximada de FLOPs para un Transformer (basado en el tamaño de entrada)
        # Nota: Si prefieres la medida exacta, se puede usar la librería externa 'thop' o 'fvcore'.
        # Para mantenerlo nativo, guardamos los parámetros como proxy de complejidad.
        return total_params

    def log_epoch(
        self, epoch: int, train_loss: float, train_acc: float, 
        test_loss: float, test_acc: float, epoch_time: float, 
        inference_time_per_sample: float, model: torch.nn.Module
    ):
        """Registers y guarda las métricas de una época en el archivo CSV."""
         
        # Medición de memoria VRAM (solo si hay GPU disponible)
        if torch.cuda.is_available():
            max_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convertir a MB
            torch.cuda.reset_peak_memory_stats() # Resetear para la siguiente época
        else:
            max_vram = 0.0 # CPU no reporta VRAM de este modo
            
        total_params = sum(p.numel() for p in model.parameters())
        
        approx_flops = "N/A" # Lo calcularemos con precisión en las fases experimentales
        
        row = [
            epoch, 
            round(train_loss, 4), round(train_acc, 2),
            round(test_loss, 4), round(test_acc, 2),
            round(epoch_time, 2), round(inference_time_per_sample, 4),
            round(max_vram, 2), approx_flops, total_params
        ]
        
        # Guardar en caliente en el disco (así si el entrenamiento se corta, no pierdes los datos)
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            
        self.history.append(row)