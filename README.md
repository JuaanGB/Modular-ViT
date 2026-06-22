# Modular-ViT

Modular-ViT es un framework de investigación desarrollado en PyTorch para Vision Transformers (ViT) con una arquitectura completamente modular. Permite intercambiar fácilmente componentes como mecanismos de tokenización, codificaciones posicionales, estrategias de agregación, módulos de atención y otros elementos del pipeline de procesamiento.

El proyecto ha sido desarrollado como parte del Trabajo Fin de Grado (TFG) del Grado en Ingeniería Informática de la Universidad de Murcia, con el objetivo de facilitar la experimentación reproducible y la comparación sistemática de distintas propuestas arquitectónicas para Vision Transformers.

---

# Características principales

- Arquitectura completamente modular.
- Configuración de experimentos mediante ficheros YAML.
- Ejecución reproducible mediante contenedores Docker.
- Comparación sencilla entre diferentes mecanismos de tokenización.
- Soporte para múltiples datasets.
- Cálculo automático de métricas como:
  - Precisión de entrenamiento y test.
  - FLOPs.
  - Número de parámetros.
  - Consumo máximo de memoria GPU.
  - Tiempo de entrenamiento.
  - Tiempo de inferencia.

---

# Estructura del proyecto

```text
Modular-ViT/
│
├── configs/            # Configuraciones YAML de experimentos
├── data/               # Definición y carga de datasets (omitida en Git)
├── docs/               # Documentación adicional
├── experiments         # Resultados experimentales
├── models/             # Componentes modulares del modelo
├── utils/              # Utilidades adicionales
├── main.py             # Punto de entrada principal
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# Requisitos

La forma recomendada de ejecución es mediante Docker.

Software utilizado durante el desarrollo:

| Software       | Versión |
| -------------- | ------- |
| Python         | 3.10.13 |
| PyTorch        | 2.2.0   |
| Docker         | 24.0.2  |
| Docker Compose | v2.18.1 |
| CUDA           | 12.1    |
| cuDNN          | 8       |

Imagen base utilizada:

```dockerfile
pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
```

Dependencias adicionales:

| Librería   | Versión            |
| ---------- | ------------------ |
| fvcore     | 0.1.5.post20221221 |
| matplotlib | >= 3.7.0           |
| numpy      | >= 1.24.0          |
| pandas     | >= 2.0.0           |
| PyYAML     | 6.0.3              |

---

# Construcción del entorno

Desde la raíz del proyecto:

```bash
docker compose build --no-cache
```

---

# Ejecución de un experimento

Ejemplo de entrenamiento utilizando una configuración YAML:

```bash
docker compose run --rm vit python main.py --config=configs/cifar10-vitbase-32x32-patch8x8.yaml
```

Cada experimento queda completamente definido por el fichero de configuración especificado.

---

# Reproducibilidad

Para garantizar la reproducibilidad de los experimentos, el framework fija una semilla aleatoria común para:

- Python
- NumPy
- PyTorch

Configuración utilizada en el TFG:

```python
set_seed(2026, deterministic=True)
```

Asimismo, todos los experimentos se ejecutan bajo la misma infraestructura hardware y software, modificando únicamente los componentes objeto de estudio.

---

# Entorno experimental utilizado en el TFG

Todos los experimentos se han ejecutado en el nodo **nikola** del GACOP Computing Cluster (Grupo de Arquitectura y Computación Paralela) de la Universidad de Murcia.

## Hardware

| Componente | Especificación               |
| ---------- | ---------------------------- |
| CPU        | Intel Core i5-10400F         |
| RAM        | 16 GB DDR4-2666 MHz (2×8 GB) |
| GPU        | NVIDIA GeForce GTX 1080      |

## Software

| Componente     | Versión |
| -------------- | ------- |
| Python         | 3.10.13 |
| PyTorch        | 2.2.0   |
| Docker         | 24.0.2  |
| Docker Compose | v2.18.1 |
| CUDA           | 12.1    |
| cuDNN          | 8       |

---

# Configuración experimental del TFG

Las evaluaciones realizadas en el Trabajo Fin de Grado emplean:

- Dataset CIFAR-10.
- Resolución original de imagen: 32×32.
- Dimensión de embedding: 192.
- 70 épocas de entrenamiento.
- Mismos hiperparámetros base para todas las comparaciones.
- Misma infraestructura hardware y software para todos los experimentos.

El objetivo es aislar el impacto de los mecanismos de tokenización y otros componentes modulares evaluados.

---

# Resultados generados

Durante la ejecución de un experimento se registran métricas como:

- Train Loss
- Test Loss
- Train Accuracy
- Test Accuracy
- FLOPs
- Número de parámetros
- Tiempo por época
- Tiempo de inferencia
- Uso máximo de memoria GPU

Los resultados pueden utilizarse posteriormente para generar tablas y gráficas comparativas.

Dentro de la carpeta `utils`, hay una herramienta [`experiments_viewer.py`](utils/experiments_viewer.py) que permite cargar y visualizar fácilmente diferentes métricas de los ficheros de resultados generados.

![](docs/Herramienta%20software%20métricas.png)

---

# Enlaces de interés

Documentación adicional disponible en la carpeta `docs/`:

- [¿Cómo agregar un nuevo mecanismo de tokenización?](docs/README_TOKENIZATION.md)
- [¿Cómo añadir un nuevo dataset?](docs/README_DATASET.md)
- [¿Cómo crear ficheros de configuración YAML?](docs/README_CONFIGS.md)

---

# Autor

Trabajo Fin de Grado en Ingeniería Informática.

Universidad de Murcia.

Curso académico 2025-2026.