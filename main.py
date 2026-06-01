import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils.loader import get_data_loaders
from utils.yaml import *
from utils.tracker import ExperimentTracker


def run_experiment(config_path: str):

    config = parse_yaml_config(config_path)
    dataset_name, batch_size, img_size = get_dataset_info(config)
    epochs, learning_rate, weight_decay = get_training_info(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Ejecutando experimento en el dispositivo: {device}")

    experiment_name = f"vanilla_vit_{dataset_name.lower()}_bs{batch_size}_lr{learning_rate}"
    tracker = ExperimentTracker(experiment_name=experiment_name, img_size=img_size, batch_size=batch_size)

    print(f"[*] Cargando el conjunto de datos: {dataset_name}...")
    train_loader, test_loader, num_classes = get_data_loaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        img_size=img_size
    )

    print("[*] Inicializando Modular Vision Transformer...")
    model = create_vit_from_config(config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"[*] Inicio del bucle de entrenamiento ({epochs} épocas)...")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        print(f"\n[>] Iniciando Época [{epoch}/{epochs}]")
        print("-" * 40)

        # =====================================================================
        # 1. TRAIN (Bucle único de entrenamiento)
        # =====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += images.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100 * train_correct / train_total

        # =====================================================================
        # 2. TEST (Evaluación en el conjunto de validación/test)
        # =====================================================================
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        test_start_time = time.time()

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)

                test_correct += (preds == targets).sum().item()
                test_total += images.size(0)

        test_end_time = time.time()
        epoch_end_time = time.time()

        epoch_test_loss = test_loss / test_total
        epoch_test_acc = 100 * test_correct / test_total

        epoch_duration = epoch_end_time - epoch_start_time
        inference_time_per_sample = ((test_end_time - test_start_time) * 1000) / test_total

        # =====================================================================
        # LOG (Quitamos la variable train_eval_acc que ya no existe)
        # =====================================================================
        tracker.log_epoch(
            epoch=epoch,
            train_loss=epoch_train_loss,
            train_acc=epoch_train_acc,
            test_loss=epoch_test_loss,
            test_acc=epoch_test_acc,
            epoch_time=epoch_duration,
            inference_time_per_sample=inference_time_per_sample,
            model=model
        )

        print(
            f"Época [{epoch}/{epochs}] ({epoch_duration:.2f}s) -> "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.2f}% || "
            f"Test Loss: {epoch_test_loss:.4f} | "
            f"Test Acc: {epoch_test_acc:.2f}% | "
            f"Inferencia: {inference_time_per_sample:.3f}ms/img"
        )

    print("-" * 70)
    print(f"[+] Experimento completado. Guardado en: {tracker.csv_path}")


if __name__ == "__main__":
    run_experiment("configs/cifar10.yaml")