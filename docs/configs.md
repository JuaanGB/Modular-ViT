# Guía de Configuración del Vision Transformer Modular

Este documento describe todos los parámetros disponibles en el fichero de configuración YAML utilizado para instanciar dinámicamente el modelo Vision Transformer modular.

---

# Estructura General

```yaml
dataset:
  ...

training:
  ...

model:
  ...
```

---

# Dataset

Configuración del conjunto de datos utilizado durante el entrenamiento.

| Parámetro | Tipo | Descripción |
|------------|--------|-------------|
| `name` | `string` | Nombre identificativo del dataset. |
| `in_channels` | `int` | Número de canales de entrada de las imágenes. Por ejemplo, `3` para RGB o `1` para escala de grises. |
| `img_size` | `int` | Tamaño de las imágenes de entrada (asumiendo imágenes cuadradas). |
| `num_classes` | `int` | Número de clases a predecir. |

### Ejemplo

```yaml
dataset:
  name: "CIFAR10"
  in_channels: 3
  img_size: 32
  num_classes: 10
```

---

# Entrenamiento

Parámetros relacionados con el proceso de optimización.

| Parámetro | Tipo | Descripción |
|------------|--------|-------------|
| `batch_size` | `int` | Número de muestras procesadas simultáneamente en cada iteración. |
| `lr` | `float` | Learning rate inicial utilizado por el optimizador. |
| `epochs` | `int` | Número total de épocas de entrenamiento. |
| `weight_decay` | `float` | Factor de regularización L2 aplicado por el optimizador. |

### Ejemplo

```yaml
training:
  batch_size: 64
  lr: 0.001
  epochs: 70
  weight_decay: 0.05
```

---

# Modelo

Parámetros globales de la arquitectura Vision Transformer.

| Parámetro | Tipo | Descripción |
|------------|--------|-------------|
| `name` | `string` | Nombre identificativo del experimento o configuración. |
| `embed_dim` | `int` | Dimensionalidad de los embeddings de los tokens. |
| `depth` | `int` | Número de bloques Transformer Encoder. |
| `num_heads` | `int` | Número de cabezas de atención multi-head. |
| `mlp_ratio` | `float` | Factor de expansión de la capa MLP interna de cada encoder. |

### Ejemplo

```yaml
model:
  name: "CIFAR10-ViT"
  embed_dim: 192
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0
```

---

# Etapas Modulares

La arquitectura está compuesta por varias etapas intercambiables:

1. Patch Embedding
2. Token Injection
3. Positional Encoding
4. Attention
5. Aggregation

Cada etapa dispone de distintos mecanismos seleccionables mediante el campo `type`.

---

# Patch Embedding

El módulo de Patch Embedding es responsable de transformar la imagen de entrada en una secuencia de tokens que posteriormente será procesada por el Transformer.

## Mecanismos disponibles

| Mecanismo (`type`) | Descripción |
|-------------------|-------------|
| `vanilla` | Implementación estándar del Vision Transformer. Divide la imagen en parches no solapados de tamaño fijo y proyecta cada parche al espacio de embeddings mediante una convolución con `kernel_size = stride = patch_size`. |
| `adaptive` | Implementación basada en Adaptive Patch Tokenization (2APT). Selecciona dinámicamente entre parches grandes y pequeños en función de la entropía local de la imagen, produciendo más tokens en regiones complejas y menos en regiones homogéneas. |
| `overlapping` | Variante con parches solapados. Mantiene la rejilla de tokenización pero amplía la región receptiva de cada parche mediante un kernel mayor que el stride. |

---

## Vanilla Patch Embedding (`type: "vanilla"`)

### Parámetros

| Parámetro | Tipo | Obligatorio | Descripción |
|------------|--------|-------------|-------------|
| `size` | `int` | Sí | Tamaño lateral de cada parche cuadrado. Debe ser mayor que 0. |

### Ejemplo

```yaml
patch_embedding:
  type: "vanilla"
  size: 4
```

### Notas

- Número de parches generado:

  `num_patches = (img_size / size)^2`

- No existe solapamiento entre parches.
- Corresponde exactamente al mecanismo de tokenización original propuesto en ViT.

---

## Adaptive Patch Tokenization (2APT) (`type: "adaptive"`)

### Parámetros

| Parámetro | Tipo | Obligatorio | Descripción |
|------------|--------|-------------|-------------|
| `size` | `int` | Sí | Tamaño base del parche pequeño. |
| `threshold` | `float` | Sí | Umbral de entropía utilizado para decidir si una región se representa mediante un parche grande o mediante cuatro parches pequeños. |
| `bins` | `int` | No | Número de bins utilizados para calcular la entropía de Shannon. Valor por defecto: `16`. |

### Ejemplo

```yaml
patch_embedding:
  type: "adaptive"
  size: 4
  threshold: 2.5
  bins: 16
```

### Funcionamiento

1. Se generan simultáneamente dos representaciones:
   - Parches pequeños de tamaño `size`.
   - Parches grandes de tamaño `2 × size`.

2. Para cada parche grande se calcula su entropía.

3. Si:

```text
entropía < threshold
```

se conserva únicamente el parche grande.

4. En caso contrario, el parche grande se sustituye por sus cuatro subparches pequeños.

### Interpretación de parámetros

| Parámetro | Efecto |
|------------|---------|
| `threshold` alto | Se generan más parches pequeños. Mayor detalle y mayor coste computacional. |
| `threshold` bajo | Se generan más parches grandes. Menor detalle y menor coste computacional. |
| `bins` alto | Estimación de entropía más fina pero más costosa. |

---

## Overlapping Patch Embedding (`type: "overlapping"`)

### Parámetros

| Parámetro | Tipo | Obligatorio | Descripción |
|------------|--------|-------------|-------------|
| `size` | `int` | Sí | Tamaño base del parche. También determina el stride de la convolución. |
| `overlap_pixels` | `int` | Sí | Número de píxeles adicionales que se extienden alrededor de cada parche para generar solapamiento. Debe ser mayor o igual que 0. |

### Ejemplo

```yaml
patch_embedding:
  type: "overlapping"
  size: 4
  overlap_pixels: 1
```

### Funcionamiento

La tokenización utiliza:

```text
stride = size
kernel_size = size + 2 × overlap_pixels
```

Por tanto, cada token observa una región mayor que la correspondiente a su posición en la rejilla.

### Interpretación de parámetros

| Parámetro | Efecto |
|------------|---------|
| `overlap_pixels = 0` | Equivalente al Patch Embedding estándar. |
| `overlap_pixels` pequeño | Introduce contexto local adicional entre parches vecinos. |
| `overlap_pixels` grande | Incrementa significativamente la redundancia entre tokens. |

### Ventajas

- Reduce la pérdida de información en los bordes de los parches.
- Incrementa el contexto local disponible para cada token.
- Puede mejorar la captura de estructuras visuales continuas.

---

---

# Token Injection

Permite añadir tokens especiales a la secuencia de entrada.

## Mecanismos disponibles

| Mecanismo (`type`) | Descripción | Parámetros específicos |
|-------------------|-------------|------------------------|
| `cls` | Añade un token CLS aprendible al inicio de la secuencia. | Ninguno |
| `none` | No añade ningún token adicional. | Ninguno |

### Ejemplo

```yaml
token_injection:
  type: "cls"
```

---

# Positional Encoding

El módulo de Positional Encoding introduce información espacial sobre la posición de cada token dentro de la imagen. Esta información permite que el Transformer diferencie entre tokens procedentes de distintas regiones espaciales.

## Mecanismos disponibles

| Mecanismo (`type`) | Descripción |
|-------------------|-------------|
| `absolute` | Codificación posicional absoluta aprendible basada en coordenadas bidimensionales discretas. |
| `axial-rope` | Rotary Positional Encoding bidimensional aplicado de forma independiente sobre los ejes horizontal y vertical. |
| `wepe` | Codificación posicional basada en la función elíptica de Weierstrass y su derivada, proyectadas posteriormente al espacio de embeddings. |

---

## Absolute 2D Positional Encoding (`type: "absolute"`)

Implementación de codificación posicional absoluta aprendible basada en coordenadas bidimensionales.

### Parámetros

Este mecanismo no requiere parámetros adicionales.

### Ejemplo

```yaml
positional_encoding:
  type: "absolute"
```

### Funcionamiento

Para cada token espacial con coordenadas `(i,j)`:

1. Se obtiene un embedding aprendible asociado a la fila `i`.
2. Se obtiene un embedding aprendible asociado a la columna `j`.
3. Ambos embeddings se concatenan para generar la representación posicional final.

El token CLS dispone además de un embedding posicional propio e independiente.

### Características

- Totalmente aprendible.
- Dependiente de la resolución utilizada durante el entrenamiento.
- Equivalente conceptualmente a la codificación utilizada en muchas implementaciones clásicas de Vision Transformer.

---

## Axial Rotary Positional Encoding (`type: "axial-rope"`)

Implementación de Rotary Positional Encoding bidimensional basada en la descomposición axial de las coordenadas espaciales.

### Parámetros

| Parámetro | Tipo | Obligatorio | Descripción |
|------------|--------|-------------|-------------|
| `base` | `float` | No | Constante utilizada para construir las frecuencias rotatorias. Valor por defecto: `10000.0`. |

### Ejemplo

```yaml
positional_encoding:
  type: "axial-rope"
  base: 10000.0
```

### Funcionamiento

Las coordenadas espaciales se separan en:

- Coordenada horizontal (`x`)
- Coordenada vertical (`y`)

A partir de ellas se generan dos conjuntos independientes de frecuencias sinusoidales:

- Frecuencias horizontales.
- Frecuencias verticales.

Estas frecuencias son utilizadas posteriormente por el mecanismo de atención para aplicar rotaciones sobre los vectores Query y Key.

### Interpretación del parámetro

| Parámetro | Efecto |
|------------|---------|
| `base` grande | Frecuencias más suaves y variaciones espaciales más lentas. |
| `base` pequeña | Frecuencias más altas y sensibilidad espacial más local. |

### Características

- No añade embeddings aprendibles.
- Introduce la información posicional directamente en la operación de atención.
- Mejor capacidad de extrapolación espacial que los embeddings absolutos.
- Escala mejor a resoluciones distintas de las utilizadas durante el entrenamiento.

---

## Weierstrass Positional Encoding (`type: "wepe"`)

Codificación posicional basada en la función elíptica de Weierstrass y su derivada.

### Parámetros

| Parámetro | Tipo | Obligatorio | Descripción |
|------------|--------|-------------|-------------|
| `alpha_u` | `float` | No | Factor de escala aplicado sobre el eje horizontal. Valor por defecto: `0.5`. |
| `alpha_v` | `float` | No | Factor de escala aplicado sobre el eje vertical. Valor por defecto: `0.5`. |
| `M_terms` | `int` | No | Número de términos considerados en dirección horizontal para truncar la red periódica. Valor por defecto: `4`. |
| `N_terms` | `int` | No | Número de términos considerados en dirección vertical para truncar la red periódica. Valor por defecto: `4`. |

### Ejemplo

```yaml
positional_encoding:
  type: "wepe"
  alpha_u: 0.5
  alpha_v: 0.5
  M_terms: 4
  N_terms: 4
```

### Funcionamiento

Para cada coordenada espacial:

1. Se normalizan las coordenadas de la rejilla.
2. Se construye un punto complejo dentro de una red periódica.
3. Se evalúa una aproximación truncada de:
   - La función de Weierstrass ℘(z).
   - Su derivada ℘'(z).
4. Se obtienen cuatro características geométricas:
   - Parte real de ℘(z)
   - Parte imaginaria de ℘(z)
   - Parte real de ℘'(z)
   - Parte imaginaria de ℘'(z)
5. Dichas características se comprimen mediante una función `tanh`.
6. Finalmente se proyectan al espacio de embeddings mediante una capa lineal aprendible.

### Interpretación de parámetros

| Parámetro | Efecto |
|------------|---------|
| `alpha_u` | Controla la escala geométrica del eje horizontal. |
| `alpha_v` | Controla la escala geométrica del eje vertical. |
| `M_terms` | Mayor precisión de la aproximación, a costa de más computación. |
| `N_terms` | Mayor precisión de la aproximación, a costa de más computación. |

### Características

- Basado en una representación analítica continua del espacio.
- Introduce relaciones espaciales complejas mediante funciones periódicas bidimensionales.
- Dispone de parámetros geométricos entrenables internos.
- Permite representar posiciones mediante una estructura matemática más rica que las codificaciones sinusoidales tradicionales.

---

# Aggregation

Obtiene una representación global a partir de los tokens finales producidos por el encoder.

## Mecanismos disponibles

| Mecanismo (`type`) | Descripción | Parámetros específicos |
|-------------------|-------------|------------------------|
| `cls` | Utiliza exclusivamente el token CLS. | Ninguno |
| `gap` | Global Average Pooling sobre todos los tokens. | Ninguno |
| `gap+gmp` | Concatenación de Global Average Pooling y Global Max Pooling. | Ninguno |

### Ejemplo

```yaml
aggregation:
  type: "cls"
```

---