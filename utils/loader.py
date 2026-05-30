import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 3. Diccionario mapeador con los datasets compatibles de torchvision
# Lista de datasets soportados: https://docs.pytorch.org/vision/main/datasets.html
dataset_mapping = {
    "MNIST": torchvision.datasets.MNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
}
 
def get_data_loaders(
    dataset_name: str, 
    batch_size: int = 64, 
    img_size: int = 224, 
    num_workers: int = 2
):
    """
    Fábrica genérica para cargar y descargar datasets de torchvision de forma parametrizada.
    Guarda los datos automáticamente en la carpeta raíz 'data/nombre_del_dataset'.
    """
    
    # 1. Definir la ruta de destino dentro de la carpeta 'data/'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', dataset_name)
    
    # 2. Transformaciones estándar necesarias para un Vision Transformer:
    # - Resize: El ViT clásico original espera imágenes de 224x224 (u otra escala fija, parçámetro img_size).
    # - ToTensor: Convierte la imagen a tensores de PyTorch y normaliza píxeles a [0, 1].
    # - Normalize: Normalización estándar de ImageNet para estabilizar el gradiente.
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(), # Aumento de datos básico para el entrenamiento
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validar que el string que pasamos exista en torchvision
    name_upper = dataset_name.upper()
    if name_upper not in dataset_mapping:
        raise ValueError(f"Dataset '{dataset_name}' no soportado. Elige entre: {list(dataset_mapping.keys())}")
        
    dataset_class = dataset_mapping[name_upper]

    # 4. Instanciar y descargar automáticamente (si no existen en disco)
    # MNIST solo tiene 1 canal (escala de grises). Para que no rompa el ViT (que espera 3),
    # hacemos un pequeño truco adaptativo usando el argumento 'transform'
    if name_upper == "MNIST":
        # Forzamos a que convierta la imagen en 3 canales RGB copiando el canal de grises
        transform_train.transforms.insert(1, transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
        transform_test.transforms.insert(1, transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    train_dataset = dataset_class(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = dataset_class(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform_test
    )

    # 5. Crear los objetos DataLoader para gestionar los mini-batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Extraemos de forma dinámica el número de clases del dataset (necesario para la capa lineal de salida del ViT)
    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
    else:
        # Fallback para datasets que usan targets directos
        num_classes = len(train_dataset.targets.unique())

    return train_loader, test_loader, num_classes