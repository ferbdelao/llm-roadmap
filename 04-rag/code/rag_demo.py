"""
Fragmento ilustrativo de un pipeline RAG mínimo, "a mano" (sin librerías
de orquestación como LangChain), para ver claramente cada paso:

    1. Generar embeddings de un conjunto de documentos
    2. Indexarlos (aquí: simplemente guardarlos en una lista/matriz)
    3. Dada una pregunta, recuperar los k documentos más similares
    4. Construir el prompt final con el contexto recuperado

Requisitos sugeridos:
    pip install sentence-transformers numpy
"""

import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class MiniVectorStore:
    """Base de datos vectorial mínima, en memoria, solo para ilustrar el concepto.
    En producción usarías FAISS, Chroma, Pinecone, etc.
    """

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, document: str, embedding: np.ndarray):
        self.documents.append(document)
        self.embeddings.append(embedding)

    def search(self, query_embedding: np.ndarray, k: int = 3):
        scores = [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(self.documents[i], scores[i]) for i in top_k_idx]


def build_rag_prompt(question: str, retrieved_docs: list[str]) -> str:
    context = "\n\n".join(retrieved_docs)
    return f"""Usa el siguiente contexto para responder la pregunta.
Si la respuesta no está en el contexto, di que no lo sabes.

Contexto:
{context}

Pregunta: {question}
Respuesta:"""


if __name__ == "__main__":
    # --- Embeddings de juguete (vectores aleatorios solo para demostrar el flujo) ---
    # En la práctica usarías, por ejemplo:
    #   from sentence_transformers import SentenceTransformer
    #   model = SentenceTransformer("all-MiniLM-L6-v2")
    #   embedding = model.encode(texto)

    np.random.seed(0)
    store = MiniVectorStore()

    documentos = [
        "El Transformer fue propuesto en el paper 'Attention Is All You Need' en 2017.",
        "LoRA reduce los parámetros entrenables usando matrices de bajo rango.",
        "RAG combina recuperación de información con generación de texto.",
    ]

    for doc in documentos:
        fake_embedding = np.random.rand(384)  # simula un embedding real de 384 dims
        store.add(doc, fake_embedding)

    pregunta = "¿Qué es RAG?"
    fake_query_embedding = np.random.rand(384)  # en la práctica: model.encode(pregunta)

    resultados = store.search(fake_query_embedding, k=2)
    documentos_recuperados = [doc for doc, score in resultados]

    prompt_final = build_rag_prompt(pregunta, documentos_recuperados)
    print(prompt_final)

    # El siguiente paso (no incluido aquí) sería enviar `prompt_final` a un LLM,
    # por ejemplo mediante la API de Hugging Face, OpenAI, o un modelo local.
