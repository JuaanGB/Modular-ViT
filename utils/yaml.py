# yaml.py
import os
import sys
import yaml

from models.vit import ModularViT

from models.patch_embedding.vanilla import VanillaPatchEmbedding
from models.patch_embedding.adaptive import APTPatchEmbedding

from models.token_injection.cls import CLSTokenInjection
from models.token_injection.none import IdentityTokenInjection

from models.positional_encoding.absolute import Absolute2DPositionalEncoding
from models.positional_encoding.wepe import Weierstrass2DPositionalEncoding

from models.encoder.encoder_block import TransformerEncoderBlock

from models.aggregation.cls import CLSAggregation
from models.aggregation.gap import GAPAggregation
from models.aggregation.gap_gmp import GAPGMPAggregation 

from models.ExecutionState import ExecutionState

import torch.nn as nn

PATCH_MECHANISMS = {
    "vanilla": VanillaPatchEmbedding,
    "adaptive": APTPatchEmbedding,
    # "cnn": CNNPatchEmbedding
}

TOKEN_INJECTION_MECHANISMS = {
    "cls": CLSTokenInjection,
    "none": IdentityTokenInjection
}

POSITIONAL_ENCODING_MECHANISMS = {
    "absolute": Absolute2DPositionalEncoding,
    "wepe": Weierstrass2DPositionalEncoding,
    # "learnable": LearnablePositionalEncoding,
}

AGGREGATION_MECHANISMS = {
    "cls": CLSAggregation,
    "gap": GAPAggregation,
    "gap+gmp": GAPGMPAggregation
    # "mean": MeanAgregation
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

    # 1. Inicialización del ExecutionState (centralización de atributos globales entre fases de tokenización)
    execution_state = ExecutionState()

    # 2. Extraer Hiperparámetros
    img_size = dataset_cfg.get("img_size", 224)
    in_channels = dataset_cfg.get("in_channels", 3)
    num_classes = dataset_cfg.get("num_classes", 10)

    embed_dim = model_cfg.get("embed_dim", 768)
    depth = model_cfg.get("depth", 12)
    num_heads = model_cfg.get("num_heads", 12)
    mlp_ratio = model_cfg.get("mlp_ratio", 4.0)

    # 3. Creación del mecanismo de PatchEmbedding
    patch_cfg = model_cfg.get("patch_embedding", {})
    patch_type = patch_cfg.get("type", "vanilla")
    
    patch_mechanism = PATCH_MECHANISMS[patch_type].create_from_config(
        config=patch_cfg, 
        img_size=img_size, 
        in_channels=in_channels, 
        embed_dim=embed_dim,
        execution_state=execution_state
    )
    print(f"\t-> Patch Embedding empleado: {patch_type}")

    # 4. Creación del mecanismo de TokenInjection
    token_cfg = model_cfg.get("token_injection", {})
    token_type = token_cfg.get("type", "cls")

    token_injection_mechanism = TOKEN_INJECTION_MECHANISMS[token_type].create_from_config(
        config=token_cfg,
        execution_state=execution_state
    )
    print(f"\t-> Token Injection empleado: {token_type}")

    # 5. Creación del mecanismo de PositionalEncoding
    pos_cfg = model_cfg.get("positional_encoding", {})
    pos_encoding_type = pos_cfg.get("type", "absolute")
    
    pos_encoding_mechanism = POSITIONAL_ENCODING_MECHANISMS[pos_encoding_type].create_from_config(
        config=pos_cfg,
        execution_state=execution_state
    )
    print(f"\t-> Positional Encoding empleado: {pos_encoding_type}")
    
    # 6. Creación dinámica de la lista de Bloques del Transformer (Encoder Blocks)
    # Aquí pasamos los hiperparámetros que leídos del YAML
    encoder_blocks = nn.ModuleList([
        TransformerEncoderBlock(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio
        )
        for _ in range(depth)
    ])

    # 7. Creación del mecanismo de Agregación (CLS o Mean Pooling)
    agg_cfg = model_cfg.get("aggregation", {})
    agg_type = agg_cfg.get("type", "cls")
    
    aggregation_mechanism = AGGREGATION_MECHANISMS[agg_type].create_from_config(
        config=agg_cfg,
        execution_state=execution_state
    )
    print(f"\t-> Aggregation empleado: {agg_type}")

    # 8. Devolvemos el ViT Modular creado
    return ModularViT(
        patch_embedding=patch_mechanism,
        token_injection=token_injection_mechanism,
        positional_encoding=pos_encoding_mechanism,
        encoder_blocks=encoder_blocks,
        aggregation=aggregation_mechanism,
        num_classes=num_classes
    )
