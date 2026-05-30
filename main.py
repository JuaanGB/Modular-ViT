import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Importamos nuestros módulos propios
from utils.loader import get_data_loaders
from models.vit import create_vanilla_vit_base
from utils.tracker import ExperimentTracker

def run_experiment(
    dataset_name: str = "CIFAR10",
    batch_size: int = 64,
    epochs: int = 5,
    learning_rate: float = 5e-4,
    img_size: int = 224
):
    # 1. Configuración del dispositivo (Garantizar uso de GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Ejecutando experimento en el dispositivo: {device}")

    # Definir el nombre del experimento para la organización de los CSVs
    experiment_name = f"vanilla_vit_{dataset_name.lower()}_bs{batch_size}_lr{learning_rate}"
    tracker = ExperimentTracker(experiment_name=experiment_name)

    # 2. Carga y descarga parametrizada del Dataset
    print(f"[*] Cargando el conjunto de datos: {dataset_name}...")
    train_loader, test_loader, num_classes = get_data_loaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        img_size=img_size
    )

    # 3. Inicialización de la Arquitectura Modular (ViT-Base original)
    print("[*] Inicializando Modular Vision Transformer (ViT-Base/16)...")
    model = create_vanilla_vit_base(img_size=img_size, num_classes=num_classes)
    model = model.to(device)

    # 4. Definición de la función de pérdida y optimizador clásico
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    print(f"[*] Inicio del bucle de entrenamiento ({epochs} épocas)...")
    print("-" * 70)

    # Bucle principal por épocas
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        print(f"\n[>] Iniciando Época [{epoch}/{epochs}]")
        print("-" * 40)
        
        # ==========================================
        # FASE DE ENTRENAMIENTO (TRAIN LOOP)
        # ==========================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        num_batches = len(train_loader)
        batch_start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Paso hacia adelante (Forward)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Paso hacia atrás y optimización (Backward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Acumulación de métricas en caliente
            train_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == targets).sum().item()
            train_total += images.size(0)
            
            # Print de progreso cada 20 lotes (para no saturar la terminal)
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                batch_loss = loss.item()
                batch_acc = ((predictions == targets).sum().item() / images.size(0)) * 100
                elapsed_batch_time = time.time() - batch_start_time
                
                print(f"   Batch [{batch_idx + 1}/{num_batches}] | "
                      f"Loss: {batch_loss:.4f} | "
                      f"Acc instantánea: {batch_acc:.2f}% | "
                      f"Tiempo lote: {elapsed_batch_time:.2f}s")
                
                batch_start_time = time.time() # Resetear reloj del lote
            
        # Calcular medias globales de entrenamiento
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        print(f"   [+] Entrenamiento finalizado. Evaluando modelo...")

        # ==========================================
        # FASE DE EVALUACIÓN (TEST LOOP)
        # ==========================================
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
                predictions = outputs.argmax(dim=1)
                test_correct += (predictions == targets).sum().item()
                test_total += images.size(0)
                
        test_end_time = time.time()
        epoch_end_time = time.time()
        
        # Calcular medias globales de test
        epoch_test_loss = test_loss / test_total
        epoch_test_acc = (test_correct / test_total) * 100
        
        # Cálculos de tiempos exactos
        epoch_duration = epoch_end_time - epoch_start_time
        total_test_time_ms = (test_end_time - test_start_time) * 1000
        inference_time_per_sample = total_test_time_ms / test_total

        # 5. Registrar y guardar las métricas en el archivo CSV
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
        
        # Mostrar progreso por consola de forma limpia
        print(f"Época [{epoch}/{epochs}] ({epoch_duration:.2f}s) -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || "
              f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}% | "
              f"Inferencia: {inference_time_per_sample:.3f}ms/img")

    print("-" * 70)
    print(f"[+] Experimento completado con éxito. Métricas guardadas en: {tracker.csv_path}")

if __name__ == "__main__":
    # Parámetros por defecto para validar el funcionamiento de la Fase 1
    # Puedes cambiar "CIFAR10" por "MNIST" si quieres una prueba ultrarrápida en CPU
    run_experiment(
        dataset_name="CIFAR10", 
        batch_size=64, 
        epochs=5, 
        learning_rate=5e-4
    )