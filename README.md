# Modular-ViT
Modular-ViT es un framework de investigación en PyTorch para Vision Transformers con componentes totalmente intercambiables, como patch embedding, codificación posicional, inyección de tokens, atención y agregación. Diseñado para la experimentación reproducible y la comparación de arquitecturas.
---
# Creación del contenedor
``docker compose build --no-cache``
# Ejecución del entrenamiento
``docker compose run --rm vit python main.py --config=configs/cifar10-vitbase-32x32-patch8x8.yaml``