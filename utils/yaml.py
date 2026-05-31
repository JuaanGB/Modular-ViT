# yaml.py
import os
import sys
import yaml
from models.vit import ModularViT
from models.patch_embedding.vanilla import VanillaPatchEmbedding
from models.token_injection.cls import CLSTokenInjection

PATCH_MECHANISMS = {
    "vanilla": VanillaPatchEmbedding,
    # "adaptive": AdaptivePatchEmbedding,
    # "cnn": CNNPatchEmbedding
}

TOKEN_INJECTION_MECHANISMS = {
    "cls": CLSTokenInjection
}

POSITIONAL_ENCODING_MECHANISMS = {
    # "learnable": LearnablePositionalEncoding,
    # "weierstrass": WeierstrassPositionalEncoding
}

def parse_yaml_config(config_path: str) -> dict:
    """Valida y parsea el fichero de configuración YAML."""
    if not os.path.exists(config_path):
        print(f"[-] ERROR: El fichero de configuración no existe en la ruta: '{config_path}'")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"[-] ERROR al parsear el archivo YAML: {exc}")
            sys.exit(1)

def get_dataset_info(config: dict):
    """
    Extrae la información clave del dataset y entrenamiento.
    Returns:
        tuple: (dataset_name, batch_size, img_size)
    """
    dataset_config = config.get("dataset", {})
    training_config = config.get("training", {})
    
    dataset_name = dataset_config.get("name", "CIFAR10")
    img_size = dataset_config.get("img_size", 224)
    batch_size = training_config.get("batch_size", 64)
    
    return dataset_name, batch_size, img_size

def get_training_info(config: dict):
    """
    Extrae la información clave para el entrenamiento y optimizador
    Returns:
        tuple: (epochs, learning_rate, weigth_decay)"""
    training_config = config.get("training", {})

    epochs = training_config.get("epochs", 0)
    learning_rate = training_config.get("lr", 0.01)
    wieght_decay = training_config.get("wieght_decay", 0.05)
    
    return epochs, learning_rate, wieght_decay
    
def create_vit_from_config(config: dict) -> ModularViT:
    """
    Instancia un ModularViT dinámicamente extrayendo la configuración del fichero YAML
    """

    # Validaciones de la estructura
    dataset_cfg = config.get("dataset", None)
    model_cfg = config.get("model", None)
    
    if dataset_cfg is None:
        print("[-] ERROR al parsear el archivo YAML: Falta el bloque obligatorio 'dataset'")
        sys.exit(1)
    if model_cfg is None:
        print("[-] ERROR al parsear el archivo YAML: Falta el bloque obligatorio 'model'")
        sys.exit(1)

    # Creación de mecanismo de PatchEmbedding
    patch_type = model_cfg.get("patch_embedding", None).get("type", None)

    img_size = dataset_cfg.get("img_size", 0)
    in_channels = dataset_cfg.get("in_channels", 0)
    embed_dim = model_cfg.get("embed_dim", 0)
    patch_mechanism = PATCH_MECHANISMS[patch_type].create_from_config(config=model_cfg.get("patch_embedding", {}), 
                                                                      img_size=img_size, 
                                                                      in_channels=in_channels, 
                                                                      embed_dim=embed_dim
    )

    # Creación de mecanismo de TokenInjection
    token_type = model_cfg.get("token_injection", None).get("type", None)

    token_injection_mechanism = TOKEN_INJECTION_MECHANISMS[token_type].create_from_config(config=model_cfg.get("token_injection", {}),
                                                                                          embed_dim=embed_dim
    )
    
    # Devolvemos el ViT Modular creado
    return ModularViT(
        patch_embedding=patch_mechanism,
        token_injection=token_injection_mechanism,
        #positional_encoding: BasePositionalEncoding,
        #encoder_blocks: nn.ModuleList, # Lista de bloques que usan ModularAttention
        #aggregation: BaseAggregation,
        #num_classes: int
    )
