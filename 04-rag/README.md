# 4. RAG (Retrieval-Augmented Generation): Dándole Memoria al Modelo

## 4.1 El problema que resuelve RAG

Un LLM preentrenado tiene dos limitaciones fundamentales:

1. **Conocimiento congelado**: solo "sabe" lo que vio durante su entrenamiento, hasta cierta fecha de corte. No conoce eventos posteriores ni datos privados de tu empresa o proyecto.
2. **Alucinaciones**: cuando no sabe algo, puede generar respuestas que *suenan* convincentes pero son incorrectas, porque su tarea fundamental es predecir texto plausible, no verificar hechos.

**RAG (Retrieval-Augmented Generation)**, propuesto por Lewis et al. (2020), resuelve esto combinando:
- Un sistema de **recuperación de información** (*retrieval*) que busca documentos relevantes en una base de conocimiento externa.
- Un LLM que **genera** la respuesta usando esos documentos como contexto adicional.

En otras palabras: en lugar de confiar únicamente en lo que el modelo memorizó durante el entrenamiento, le damos la información relevante directamente en el *prompt*, en el momento de la consulta.

## 4.2 El pipeline de RAG, paso a paso

```
Pregunta del usuario
        │
        ▼
 1. Embedding de la pregunta
        │
        ▼
 2. Búsqueda de similitud en una base vectorial
        │
        ▼
 3. Recuperación de los k documentos más relevantes
        │
        ▼
 4. Construcción del prompt: pregunta + documentos recuperados
        │
        ▼
 5. El LLM genera la respuesta usando ese contexto
```

### Paso 1-2: Embeddings y bases de datos vectoriales

Recordando el módulo 1: cada fragmento de texto (documento, párrafo, "chunk") se convierte en un vector mediante un modelo de embeddings (no necesariamente el mismo LLM que genera la respuesta final — suele usarse un modelo especializado y más pequeño, como `sentence-transformers`).

Estos vectores se almacenan en una **base de datos vectorial** (Chroma, Pinecone, Weaviate, FAISS, etc.), optimizada para búsquedas de similitud (k-NN aproximado) en espacios de alta dimensión.

### Paso 3: Recuperación (Retrieval)

Dada la pregunta del usuario, se calcula su embedding y se buscan los $k$ vectores más cercanos (usando similitud coseno o distancia euclidiana) en la base vectorial:

$$\text{top-}k = \underset{d \in \mathcal{D}}{\text{argmax}_k} \; \cos(\text{embed}(q), \text{embed}(d))$$

donde $q$ es la pregunta y $\mathcal{D}$ es la colección de documentos (o fragmentos de documentos) indexados.

### Paso 4-5: Generación aumentada

Los documentos recuperados se insertan en el *prompt* del LLM, típicamente con una plantilla como:

```
Usa el siguiente contexto para responder la pregunta. Si la respuesta no está
en el contexto, di que no lo sabes.

Contexto:
{documentos_recuperados}

Pregunta: {pregunta_usuario}
Respuesta:
```

El LLM genera entonces la respuesta condicionando su predicción tanto en su conocimiento preentrenado como en el contexto recién inyectado.

## 4.3 Chunking: una decisión de diseño clave

Los documentos largos deben dividirse en fragmentos ("chunks") antes de generar embeddings, porque:
- Los modelos de embeddings tienen un límite de tokens de entrada.
- Fragmentos más pequeños y enfocados producen embeddings más precisos que documentos completos muy largos y heterogéneos.

Estrategias comunes:
- **Tamaño fijo** con *overlap* (p. ej. 500 tokens con 50 de solapamiento, para no perder contexto en los bordes).
- **Por estructura semántica** (párrafos, secciones, encabezados).

## 4.4 RAG vs. Fine-Tuning: ¿cuándo usar cada uno?

| | RAG | Fine-Tuning |
|---|---|---|
| Actualizar conocimiento | Fácil: solo actualizas la base de datos | Requiere reentrenar |
| Costo computacional | Bajo (no se reentrena el modelo) | Alto (incluso con LoRA, requiere GPU y datos etiquetados) |
| Cambiar el *estilo*/*tono* del modelo | Limitado | Ideal |
| Citar fuentes / trazabilidad | Natural (sabes qué documento se usó) | No directamente |
| Conocimiento muy específico y cambiante | Ideal | No ideal |

En la práctica, muchos sistemas en producción combinan ambos: un modelo afinado (fine-tuned) para el tono/dominio correcto, alimentado con RAG para información actualizada y verificable.

## Código

Ver [`code/rag_demo.py`](./code/rag_demo.py) para un fragmento ilustrativo de un pipeline RAG mínimo: generar embeddings, indexarlos, recuperar los más relevantes y construir el prompt final.

## Recursos recomendados

- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (paper original de RAG).
- [Documentación de LangChain — RAG](https://python.langchain.com/docs/tutorials/rag/)
- [Documentación de LlamaIndex](https://docs.llamaindex.ai/)
