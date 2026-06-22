# Cómo añadir un nuevo dataset

Este documento describe el procedimiento para añadir nuevos datasets al framework Modular-ViT.

---

# Requisitos

Actualmente el framework utiliza la API de datasets de TorchVision.

Por tanto, únicamente pueden utilizarse datasets disponibles en la documentación oficial de TorchVision, concretamente los relacionados con problemas de clasificación:

[TorchVision Datasets Documentation](https://docs.pytorch.org/vision/main/datasets.html)

Antes de añadir un nuevo dataset, compruebe que existe una implementación oficial dentro de `torchvision.datasets`.

*NOTA: Algunos datasets podrían requerir de modificaciones adicionales en el código de `loader.py` si no disponen de un conjunto de entrenamiento y test preparados.*

---

# Funcionamiento interno

La carga de datasets se realiza mediante el fichero:

```text
code/utils/loader.py
```

Este fichero contiene un diccionario denominado `dataset_mapping` que relaciona un nombre de dataset con su implementación correspondiente de TorchVision.

Ejemplo:

```python
dataset_mapping = {
    "MNIST": torchvision.datasets.MNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
}
```

Cuando se ejecuta un experimento, el framework utiliza este diccionario para localizar la clase correspondiente y descargar automáticamente el dataset si todavía no existe en disco.

---

# Paso 1: Verificar que el dataset existe en TorchVision

Supongamos que queremos añadir el dataset SVHN.

Primero comprobamos que existe dentro de TorchVision:

```python
torchvision.datasets.SVHN
```

Si la clase existe, el dataset es compatible con el framework.

---

# Paso 2: Añadir el dataset al mapa

Abra el fichero:

```text
code/utils/loader.py
```

y añada una nueva entrada al diccionario `dataset_mapping`.

Ejemplo:

```python
dataset_mapping = {
    "MNIST": torchvision.datasets.MNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
    "SVHN": torchvision.datasets.SVHN,
}
```

Con esta modificación el dataset ya podrá ser seleccionado desde los ficheros de configuración.

---

# Paso 3: Crear la configuración YAML

Una vez añadido al mapa, debe indicarse correctamente en el fichero de configuración.

Ejemplo:

```yaml
dataset:
  name: "SVHN"
  in_channels: 3
  img_size: 32
  num_classes: 10
```

Los parámetros deben coincidir con las características reales del dataset.

---

# Descarga automática

No es necesario descargar manualmente los datos.

Durante la primera ejecución, el framework descargará automáticamente el dataset en:

```text
data/<nombre_dataset>/
```

Por ejemplo:

```text
data/CIFAR10/
data/CIFAR100/
data/MNIST/
```

Las ejecuciones posteriores reutilizarán los datos ya almacenados.