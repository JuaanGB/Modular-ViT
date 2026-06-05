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
    )

    return flops