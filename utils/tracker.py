import os
import csv
import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FLOP_COUNTER = True
    print("[Tracker] Utilizando profiler para contar FLOPS")
except ImportError:
    HAS_FLOP_COUNTER = False


class ExperimentTracker:
    """
    Módulo encargado de registrar las métricas de rendimiento de cada experimento,
    calcular el coste computacional y persistir los resultados en formato CSV.
    """

    def __init__(self, experiment_name: str, base_dir: str = "experiments", batch_size: int = 64, img_size: int = 224):
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.img_size = img_size

        self.output_dir = base_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.csv_path = os.path.join(self.output_dir, experiment_name + ".csv")

        self.history = []

        self.headers = ["epoch", "train_loss", "train_acc",
            "test_loss", "test_acc", "epoch_time_sec",
            "inference_time_ms_per_sample", "max_vram_mb",
            "model_flops", "model_params"
        ]

        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def compute_model_stats(self, model: nn.Module, img_size: int):
        """
        Calcula parámetros y FLOPs reales usando fvcore.
        Los FLOPs se calculan por imagen.
        """

        total_params = sum(p.numel() for p in model.parameters())

        if not HAS_FLOP_COUNTER:
            return total_params, "N/A"

        device = next(model.parameters()).device

        example_input = torch.randn(1, 3, img_size, img_size, device=device)

        was_training = model.training

        try:
            model.eval()

            with torch.no_grad():
                flops = FlopCountAnalysis(model, example_input).total()

        except Exception as e:
            print(f"[ExperimentTracker] Error calculando FLOPs: {e}")
            flops = None

        finally:
            if was_training:
                model.train()

        if flops is None: flops_str = "N/A"
        elif flops >= 1e12: flops_str = f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9: flops_str = f"{flops / 1e9:.2f} GFLOPs"
        else: flops_str = f"{flops / 1e6:.2f} MFLOPs"

        return total_params, flops_str

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, test_loss: float,
        test_acc: float, epoch_time: float, inference_time_per_sample: float, model: nn.Module
    ):
        """
        Registra y guarda las métricas de una época.
        """

        if (torch.cuda.is_available() and next(model.parameters()).is_cuda):
            max_vram = (torch.cuda.max_memory_allocated() / (1024 ** 2))
            torch.cuda.reset_peak_memory_stats()
        else:
            max_vram = 0.0

        total_params, flops = self.compute_model_stats(model, self.img_size)

        row = [
            epoch,
            round(train_loss, 4),
            round(train_acc, 2),
            round(test_loss, 4),
            round(test_acc, 2),
            round(epoch_time, 2),
            round(inference_time_per_sample, 4),
            round(max_vram, 2),
            flops,
            total_params
        ]

        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()

        self.history.append(row)