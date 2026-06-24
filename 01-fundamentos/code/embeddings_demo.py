"""
Fragmento ilustrativo: embeddings y similitud coseno.

Este script NO entrena un modelo desde cero (eso tomaría horas y un corpus
grande). En su lugar, muestra cómo se manipulan embeddings en la práctica
usando vectores preentrenados, y cómo calcular la similitud coseno entre
ellos a mano (para reforzar la intuición matemática del módulo).

Requisitos sugeridos:
    pip install gensim numpy
"""

import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores.

    cos(u, v) = (u . v) / (||u|| * ||v||)
    """
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return dot_product / (norm_u * norm_v)


# --- Ejemplo con vectores "de juguete" (3 dimensiones, solo para ilustrar) ---
# En la práctica los embeddings reales tienen entre 100 y varios miles de
# dimensiones, y se obtienen entrenando un modelo o cargando uno preentrenado.

embeddings_toy = {
    "gato":     np.array([0.9, 0.1, 0.0]),
    "perro":    np.array([0.85, 0.15, 0.05]),
    "ecuacion": np.array([0.0, 0.05, 0.95]),
}

print("Similitud gato-perro: ", cosine_similarity(embeddings_toy["gato"], embeddings_toy["perro"]))
print("Similitud gato-ecuacion: ", cosine_similarity(embeddings_toy["gato"], embeddings_toy["ecuacion"]))


# --- Ejemplo con embeddings reales preentrenados (gensim) ---
# Descomenta para correrlo; descarga ~66MB de vectores la primera vez.
#
# import gensim.downloader as api
# wv = api.load("glove-wiki-gigaword-100")  # embeddings GloVe de 100 dimensiones
#
# print(wv.most_similar("gato", topn=5))
# print(cosine_similarity(wv["rey"] - wv["hombre"] + wv["mujer"], wv["reina"]))
