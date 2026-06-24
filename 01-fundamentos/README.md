# 1. Fundamentos: Vectores, Embeddings y NLP Clásico

Antes de entender un Transformer necesitas entender **cómo se representa el lenguaje numéricamente**. Este módulo cubre el puente entre "palabras" y "matemáticas".

## 1.1 ¿Por qué necesitamos números para representar palabras?

Una red neuronal solo entiende números. El primer reto del NLP (*Natural Language Processing*) es convertir texto en vectores que preserven, en lo posible, el significado.

### Tokenización

Antes de vectorizar, hay que partir el texto en unidades manejables llamadas **tokens**. No siempre son palabras completas: los LLMs modernos usan **subword tokenization** (como *Byte Pair Encoding*, BPE), que divide palabras poco frecuentes en fragmentos más pequeños.

Ejemplo:
```
"matemáticas aplicadas" → ["mate", "máticas", "Ġaplic", "adas"]
```

Esto permite manejar palabras nunca vistas combinando fragmentos conocidos, y mantiene el vocabulario en un tamaño razonable (decenas de miles de tokens en lugar de millones de palabras posibles).

## 1.2 Representaciones vectoriales: de One-Hot a Embeddings

### One-Hot Encoding

La forma más simple: cada palabra es un vector con un único 1 y el resto ceros, de tamaño igual al vocabulario $V$.

$$\text{"gato"} \rightarrow [0, 0, 1, 0, \dots, 0] \in \mathbb{R}^{|V|}$$

**Problema**: estos vectores no capturan ninguna relación semántica. "gato" y "perro" están tan lejos como "gato" y "ecuación", porque la distancia coseno entre cualquier par de vectores *one-hot* distintos es siempre la misma.

### Embeddings

Un **embedding** es una representación densa de baja dimensión ($d \ll |V|$, típicamente entre 100 y unos cuantos miles) donde palabras con significados similares quedan **cerca** en el espacio vectorial.

$$\text{"gato"} \rightarrow [0.21, -0.43, 0.85, \dots] \in \mathbb{R}^{d}$$

La intuición clave (la **hipótesis distribucional**): *"una palabra se conoce por la compañía que frecuenta"* (Firth, 1957). Palabras que aparecen en contextos similares tienden a tener significados similares.

### Word2Vec: la idea seminal

Word2Vec (Mikolov et al., 2013) aprende embeddings entrenando una red simple para predecir palabras de contexto (modelo *Skip-gram*) o predecir la palabra central a partir de su contexto (*CBOW*).

La probabilidad de que la palabra de contexto $w_c$ aparezca dado el centro $w_t$ se modela con un softmax sobre el producto punto de sus embeddings:

$$P(w_c \mid w_t) = \frac{\exp(v_{w_c}^\top v_{w_t})}{\sum_{w \in V} \exp(v_{w}^\top v_{w_t})}$$

No necesitas implementar Word2Vec desde cero hoy en día, pero esta fórmula es importante: **el mismo patrón (producto punto + softmax) reaparece en el corazón de la atención de los Transformers** (módulo 2).

### Propiedades geométricas interesantes

Una de las observaciones más famosas de los embeddings es que las relaciones semánticas se comportan como operaciones vectoriales:

$$v_{\text{rey}} - v_{\text{hombre}} + v_{\text{mujer}} \approx v_{\text{reina}}$$

Esto sugiere que el espacio de embeddings codifica relaciones lineales de significado, no solo cercanía.

## 1.3 Similitud entre vectores

Para comparar dos embeddings $u, v \in \mathbb{R}^d$ se usa típicamente la **similitud coseno**:

$$\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert}$$

Valores cercanos a 1 indican vectores muy similares (apuntan casi en la misma dirección); valores cercanos a 0, vectores poco relacionados.

## 1.4 N-gramas y modelos de lenguaje clásicos

Antes de las redes neuronales, los modelos de lenguaje estimaban la probabilidad de una secuencia de palabras usando estadística pura de **n-gramas**: contar cuántas veces aparece una secuencia de $n$ palabras en un corpus.

$$P(w_1, \dots, w_n) \approx \prod_{i=1}^{n} P(w_i \mid w_{i-(k-1)}, \dots, w_{i-1})$$

con $k$ el tamaño del n-grama (bigrama: $k=2$, trigrama: $k=3$, etc.)

**Limitación clave**: la "maldición de la dimensionalidad". Con vocabularios grandes, la mayoría de combinaciones de $n$ palabras nunca aparecen en el corpus de entrenamiento, dando probabilidad cero a secuencias perfectamente válidas. Esto es exactamente lo que los embeddings y las redes neuronales vinieron a resolver: generalizar más allá de lo visto exactamente.

## 1.5 ¿Qué sigue?

Con vectores que representan significado, el siguiente problema es: **¿cómo logra un modelo entender el significado de una palabra en relación con las demás palabras de la oración, sin importar qué tan lejos estén?** Esa es exactamente la pregunta que resuelve la arquitectura Transformer, tema del [módulo 2](../02-transformer/README.md).

## Código

Ver [`code/embeddings_demo.py`](./code/embeddings_demo.py) para un fragmento ilustrativo de:
- Cómo se ve un embedding entrenado (usando `gensim` o vectores preentrenados).
- Cómo calcular similitud coseno entre palabras.

## Recursos recomendados

- Jurafsky & Martin, *Speech and Language Processing* (capítulos de n-gramas y vectores semánticos) — disponible gratis online.
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* (paper original de Word2Vec).
- [Visualización interactiva de embeddings — TensorFlow Projector](https://projector.tensorflow.org/)
