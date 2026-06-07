"""
Custom FLOP handlers para fvcore.

Motivación
----------
fvcore no soporta algunos operadores ATen utilizados por Vision Transformers,
como aten::embedding, aten::softmax o aten::gelu.

Este fichero añade aproximaciones razonables para dichos operadores con el
objetivo de obtener una estimación más completa del coste computacional.

Notas
-----
- Los FLOPs reportados por distintos artículos no siempre consideran las mismas
  operaciones.
- Muchos trabajos de ViT ignoran softmax, GELU y operaciones element-wise.
- Por tanto, los resultados obtenidos usando estos handlers pueden ser
  ligeramente superiores a los publicados en algunos papers.
"""

from fvcore.nn import FlopCountAnalysis


def _numel_from_value(value):
    """
    Obtiene el número total de elementos de un tensor TorchScript.
    """

    try:
        sizes = value.type().sizes()

        if sizes is None:
            return 0

        numel = 1
        for s in sizes:
            if s is None:
                return 0
            numel *= s

        return numel

    except Exception:
        return 0


# ------------------------------------------------------------------
# Element-wise operators
# ------------------------------------------------------------------

def add_flops(inputs, outputs):
    """
    aten::add

    Cada suma escalar se considera 1 FLOP.

    z = x + y

    FLOPs = número de elementos del tensor de salida
    """

    return _numel_from_value(outputs[0])


def mul_flops(inputs, outputs):
    """
    aten::mul

    Cada multiplicación escalar se considera 1 FLOP.

    z = x * y

    FLOPs = número de elementos del tensor de salida
    """

    return _numel_from_value(outputs[0])


def div_flops(inputs, outputs):
    """
    aten::div

    Cada división elemental se considera 1 FLOP (a nivel teórico de coste,
    aunque en hardware suele ser más costosa que una multiplicación).

    z = x / y

    FLOPs = número de elementos del tensor de salida
    """
    return _numel_from_value(outputs[0])


def rsub_flops(inputs, outputs):
    """
    aten::rsub (Reverse Subtraction)

    Calcula: z = alpha * y - x (frecuentemente usado como 1 - x).
    Suele costar 1 FLOP (si alpha=1) o 2 FLOPs (si hay multiplicación).
    Para perfiles de ViT, contar una operación elemental (1 FLOP) es el estándar.

    FLOPs = número de elementos del tensor de salida
    """
    return _numel_from_value(outputs[0])


# ------------------------------------------------------------------
# Trigonometric operators (Frecuentes en RoPE / Positional Embeddings)
# ------------------------------------------------------------------

def cos_flops(inputs, outputs):
    """
    aten::cos

    Las funciones trascendentes varían según el hardware, pero en perfiles de 
    redes neuronales se aproximan comúnmente como 1 FLOP elemental por elemento.

    FLOPs = número de elementos del tensor de salida
    """
    return _numel_from_value(outputs[0])


def sin_flops(inputs, outputs):
    """
    aten::sin

    Similar a aten::cos, se cuenta como 1 FLOP por elemento.

    FLOPs = número de elementos del tensor de salida
    """
    return _numel_from_value(outputs[0])


# ------------------------------------------------------------------
# Padding & Data Manipulation
# ------------------------------------------------------------------

def pad_flops(inputs, outputs):
    """
    aten::pad

    El padding consiste en copiar elementos a un tensor más grande y rellenar 
    con ceros o valores constantes. Al no haber operaciones aritméticas, 
    se considera coste 0 en FLOPs (aunque penalice el ancho de banda de memoria).

    FLOPs = 0
    """
    return 0


# ------------------------------------------------------------------
# Embedding
# ------------------------------------------------------------------

def embedding_flops(inputs, outputs):
    """
    aten::embedding

    Una embedding es esencialmente una operación de lookup sobre una tabla
    de parámetros.

    No existe computación aritmética significativa asociada al acceso,
    por lo que la mayoría de trabajos consideran:

        FLOPs = 0

    aunque sí implica coste de memoria.
    """

    return 0


# ------------------------------------------------------------------
# Softmax
# ------------------------------------------------------------------

def softmax_flops(inputs, outputs):
    """
    aten::softmax

    Para cada elemento:

        exp(x)
        sum(exp(x))
        división por la suma

    Aproximación habitual:

        ~5 FLOPs por elemento

    (exp + acumulación + normalización)

    No pretende modelar exactamente la implementación hardware.
    """
    numel = _numel_from_value(outputs[0])

    return 5 * numel

# ------------------------------------------------------------------
# Reducciones (Pooling)
# ------------------------------------------------------------------

def mean_flops(inputs, outputs):
    """
    aten::mean

    Calcula la media aritmética a lo largo de una o varias dimensiones.
    
    Cómputo aproximado:
      - Sumas: 1 FLOP por cada elemento del tensor de entrada (para acumular).
      - Divisiones: 1 FLOP por cada elemento del tensor de salida (para promediar).
      
    FLOPs = numel(inputs) + numel(outputs)
    """
    # inputs[0] es el tensor de entrada que se va a promediar
    numel_in = _numel_from_value(inputs[0])
    # outputs[0] es el tensor resultante tras el pooling
    numel_out = _numel_from_value(outputs[0])
    
    return numel_in + numel_out


# ------------------------------------------------------------------
# GELU
# ------------------------------------------------------------------

def gelu_flops(inputs, outputs):
    """
    aten::gelu

    GELU exacta:

        x * Phi(x)

    donde Phi es la CDF de una gaussiana.

    Las implementaciones prácticas utilizan aproximaciones basadas en tanh.

    Diversos contadores de FLOPs para Transformers utilizan valores entre
    6 y 10 FLOPs por elemento.

    Utilizamos:

        8 FLOPs por elemento

    como compromiso razonable.
    """
    numel = _numel_from_value(outputs[0])

    return 8 * numel

# ------------------------------------------------------------------
# Operadores matemáticos adicionales
# ------------------------------------------------------------------

def pow_flops(inputs, outputs):
    """
    aten::pow (Exponenciación elemental)

    Calcula: z = x^y elemento a elemento. En hardware, elevar a una potencia
    depende de si el exponente es entero o flotante (a menudo mapeado usando logs/exps).
    Al igual que GELU o Softmax, se adopta un compromiso estándar para perfiles de ViT.

    Aproximación habitual:
        - Si es un cuadrado (x^2), equivale a 1 mul (1 FLOP).
        - Si es potencia general, puede costar varias operaciones.
    Como norma general y simplificada para perfiles, se cuenta como 1 FLOP por elemento.

    FLOPs = número de elementos del tensor de salida
    """
    return _numel_from_value(outputs[0])


def reciprocal_flops(inputs, outputs):
    """
    aten::reciprocal (Inverso elemental)

    Calcula: z = 1 / x elemento a elemento.
    Al igual que aten::div, se contabiliza como 1 operación aritmética elemental.

    FLOPs = número de elementos del tensor de salida
    """
    return _numel_from_value(outputs[0])


def neg_flops(inputs, outputs):
    """
    aten::neg (Negación aritmética)

    Calcula: z = -x elemento a elemento.
    Consiste en invertir el bit de signo. En el conteo teórico de FLOPs de alto nivel,
    los cambios de signo elementales se consideran como 1 FLOP aritmético.

    FLOPs = número de elementos del tensor de salida
    """
    return _numel_from_value(outputs[0])


# ------------------------------------------------------------------
# Manipulación de datos (Frecuente en RoPE antiguos / Atención)
# ------------------------------------------------------------------

def repeat_interleave_flops(inputs, outputs):
    """
    aten::repeat_interleave

    Repite elementos a lo largo de una dimensión. Al igual que aten::pad o 
    operaciones como reshape/permute, no realiza cálculos aritméticos flotantes, 
    sino únicamente copias y reordenamientos en memoria (overhead de ancho de banda).

    Por convención en fvcore y benchmarks de ViT:
        FLOPs = 0
    """
    return 0

# ------------------------------------------------------------------
# Reducciones y Matemáticas adicionales (Para Entropía 2APT)
# ------------------------------------------------------------------

def sum_flops(inputs, outputs):
    """
    aten::sum

    Calcula la suma acumulada a lo largo de una o varias dimensiones.
    Para reducir N elementos a 1, se requieren exactamente N - 1 sumas flotantes.
    
    Aproximación estándar:
        FLOPs = numel(inputs) - numel(outputs)
    """
    numel_in = _numel_from_value(inputs[0])
    numel_out = _numel_from_value(outputs[0])
    
    # Aseguramos que no devuelva negativo por si acaso hay dimensiones vacías
    return max(0, numel_in - numel_out)


def log2_flops(inputs, outputs):
    """
    aten::log2

    Logaritmo elemental en base 2 elemento a elemento (usado en la entropía de Shannon).
    Al ser una operación matemática trascendente sobre el tensor, se cuenta como 
    1 FLOP por elemento del tensor resultante.

    FLOPs = número de elementos del tensor de salida
    """
    return _numel_from_value(outputs[0])


def lt_flops(inputs, outputs):
    """
    aten::lt (Less Than)

    Operador lógico de comparación elemental (x < y) utilizado para generar 
    la máscara booleana de los parches basada en el umbral (threshold).
    Las operaciones lógicas/comparaciones no realizan aritmética flotante,
    por lo que en perfiles de ViT clásicos computan como 0 FLOPs.

    FLOPs = 0
    """
    return 0

def rsqrt_flop_jit(inputs, outputs):
    """Contador de FLOPs aproximado para raíces cuadradas (sqrt).
    Comúnmente se asigna 1 o 2 FLOPs por elemento."""
    output_shape = outputs[0].type().sizes()
    num_elements = 1
    for dim in output_shape:
        num_elements *= dim
    # Se estima un coste aproximado de 2 FLOPs por elemento para operaciones element-wise de raíz
    return num_elements * 2

def tanh_flop_jit(inputs, outputs):
    """Contador de FLOPs para funciones de activación trascendentales (tanh).
    Suelen aproximarse comercialmente entre 5 y 8 FLOPs por elemento."""
    output_shape = outputs[0].type().sizes()
    num_elements = 1
    for dim in output_shape:
        num_elements *= dim
    return num_elements * 6

def softplus_flop_jit(inputs, outputs):
    """Contador de FLOPs para Softplus: log(1 + exp(x)).
    Involucra una exponencial (aprox 6 FLOPs), una suma (1 FLOP) y un logaritmo (aprox 6 FLOPs)."""
    output_shape = outputs[0].type().sizes()
    num_elements = 1
    for dim in output_shape:
        num_elements *= dim
    return num_elements * 13

def atan2_flop_jit(inputs, outputs):
    """Contador de FLOPs para atan2(y, x).
    Es una función trigonométrica inversa por elemento, estimada comúnmente en unos 12 FLOPs."""
    output_shape = outputs[0].type().sizes()
    num_elements = 1
    for dim in output_shape:
        num_elements *= dim
    return num_elements * 12

def sub_flop_jit(inputs, outputs):
    """Contador de FLOPs para la resta estándar (sub).
    1 FLOP básico por cada elemento resultante del tensor."""
    output_shape = outputs[0].type().sizes()
    num_elements = 1
    for dim in output_shape:
        num_elements *= dim
    return num_elements * 1

# ------------------------------------------------------------------
# Registro
# ------------------------------------------------------------------

def add_custom_flop_handlers(flops: FlopCountAnalysis):
    """
    Registra todos los operadores personalizados.
    """
    flops.set_op_handle(
        "aten::embedding", embedding_flops,
        "aten::add", add_flops,
        "aten::mul", mul_flops,
        "aten::div", div_flops,
        "aten::rsub", rsub_flops,
        "aten::cos", cos_flops,
        "aten::sin", sin_flops,
        "aten::pad", pad_flops,
        "aten::softmax", softmax_flops,
        "aten::gelu", gelu_flops,
        "aten::mean", mean_flops,
        "aten::pow", pow_flops,
        "aten::reciprocal", reciprocal_flops,
        "aten::neg", neg_flops,
        "aten::repeat_interleave", repeat_interleave_flops,
        "aten::sum", sum_flops,
        "aten::log2", log2_flops,
        "aten::lt", lt_flops,
        "aten::sqrt", rsqrt_flop_jit,
        "aten::tanh", tanh_flop_jit,
        "aten::softplus", softplus_flop_jit,
        "aten::atan2", atan2_flop_jit,
        "aten::sub", sub_flop_jit,
    )

    return flops