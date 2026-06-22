# Cómo añadir nuevos mecanismos

Este documento describe el procedimiento para extender Modular-ViT mediante la incorporación de nuevos mecanismos de tokenización, codificación posicional, inyección de tokens o agregación.

La arquitectura del framework está diseñada siguiendo un enfoque modular basado en herencia. Cada etapa del pipeline dispone de una clase base abstracta que define la interfaz común que deben implementar todos los mecanismos compatibles.

---

# Arquitectura modular

Actualmente existen cuatro etapas intercambiables:

1. Patch Embedding
2. Token Injection
3. Positional Encoding
4. Aggregation

Cada etapa dispone de una clase base específica:

```text
models/patch_embedding/base.py      -> BasePatchEmbedding
models/token_injection/base.py      -> BaseTokenInjection
models/positional_encoding/base.py  -> BasePositionalEncoding
models/aggregation/base.py          -> BaseAggregation
```

Todas ellas heredan de `torch.nn.Module`.

---

# Métodos obligatorios

Cualquier mecanismo nuevo debe implementar dos métodos fundamentales.

## forward()

Método heredado de `nn.Module`.

Contiene la lógica principal del mecanismo y define cómo se transforman los datos durante la ejecución del modelo.

Ejemplos:

* Generar tokens a partir de una imagen.
* Añadir tokens especiales.
* Incorporar información posicional.
* Agregar la secuencia final de tokens.

La firma exacta dependerá de la etapa que se esté implementando.

---

## create_from_config()

Método estático encargado de construir la instancia a partir de la configuración YAML.

Su objetivo es:

1. Leer los parámetros específicos del mecanismo.
2. Validarlos.
3. Instanciar la clase correspondiente.

De esta forma, toda la creación de componentes se realiza automáticamente a partir del fichero de configuración.

---

# Ejemplo conceptual

Supongamos que queremos crear un nuevo mecanismo de Patch Embedding.

La nueva clase deberá heredar de:

```python
class MyPatchEmbedding(BasePatchEmbedding):
    ...
```

e implementar:

```python
def forward(self, x):
    ...
```

y

```python
@staticmethod
def create_from_config(config, ...):
    ...
```

El método `forward()` contendrá la lógica de tokenización.

El método `create_from_config()` extraerá los parámetros definidos en el YAML y construirá la instancia correspondiente.

---

# Registro del nuevo mecanismo

Una vez implementada la clase, debe registrarse en el sistema de carga dinámica.

Esto se realiza en:

```text
code/utils/yaml.py
```

Este fichero contiene varios diccionarios que relacionan una cadena del YAML con la clase concreta que debe instanciarse.

Por ejemplo:

```python
PATCH_MECHANISMS = {
    "vanilla": VanillaPatchEmbedding,
    "adaptive": TwoAPTPatchEmbedding,
    "overlapping": OverlappingPatchEmbedding,
    "cnn": CNNPatchEmbedding
}
```

Cada fase de la tokenización contiene un diccionario con nombre análogo.

---

# Añadir un nuevo Patch Embedding

Supongamos que hemos creado:

```python
class MyPatchEmbedding(BasePatchEmbedding):
    ...
```

Debemos:

1. Importar la clase en `yaml.py`.

```python
from models.patch_embedding.my_patch import MyPatchEmbedding
```

2. Añadirla al diccionario correspondiente.

```python
PATCH_MECHANISMS = {
    ...
    "my-patch": MyPatchEmbedding
}
```

---

# Uso desde YAML

Una vez registrado, el mecanismo puede utilizarse directamente desde cualquier fichero de configuración.

Ejemplo:

```yaml
model:
  patch_embedding:
    type: "my-patch"
```

Los parámetros adicionales definidos por el mecanismo serán recibidos automáticamente por su método `create_from_config()`.

---

# Resumen

Para añadir un nuevo mecanismo únicamente es necesario:

1. Elegir la etapa correspondiente.
2. Heredar de la clase base adecuada.
3. Implementar `forward()`.
4. Implementar `create_from_config()`.
5. Registrar la clase en `code/utils/yaml.py`.
6. Utilizar el nuevo identificador desde un fichero YAML.

No es necesario modificar ninguna otra parte del framework. 

*NOTA: Todas las clase base (`BasePatchEmbedding`, `BasePositionalEncoding`, `BaseTokenInjection` y `BaseAggregation`) contienen los dos métodos `forward` y `create_from_config`. Aunque este ejemplo haya sido para la fase de Patch Embedding, se puede aplicar con el resto de fases de la representación de la entrada*.
