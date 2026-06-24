# 2. El Transformer: Mecanismos de Atención

El paper que lo cambió todo: *"Attention Is All You Need"* (Vaswani et al., 2017). Antes de los Transformers, el NLP dependía de redes recurrentes (RNN, LSTM) que procesaban texto palabra por palabra, en orden, lo cual era lento y le costaba relacionar palabras lejanas entre sí. El Transformer resuelve esto procesando toda la secuencia **en paralelo** y dejando que el modelo decida, mediante un mecanismo de **atención**, qué palabras son relevantes para entender cada otra palabra.

## 2.1 La idea central: Self-Attention

La pregunta que resuelve la atención es: *dado un token, ¿qué tanto debería "fijarse" en cada uno de los demás tokens de la secuencia para entender su significado en este contexto?*

Por ejemplo, en la oración:

> "El banco estaba cerrado porque no había suficiente dinero"

La palabra "banco" necesita "fijarse" en "dinero" para resolver que se refiere a una institución financiera y no a un asiento.

### Queries, Keys y Values

Cada token se proyecta en tres vectores distintos mediante matrices de pesos aprendidas $W^Q, W^K, W^V$:

- **Query ($Q$)**: "¿qué estoy buscando?"
- **Key ($K$)**: "¿qué tengo para ofrecer?"
- **Value ($V$)**: "¿qué información contengo realmente?"

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

donde $X \in \mathbb{R}^{n \times d}$ es la matriz de embeddings de entrada ($n$ tokens, $d$ dimensiones).

### La fórmula de Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Desglosando cada pieza:

1. **$QK^\top$**: producto punto entre cada query y cada key. Mide qué tan "compatibles" son dos tokens (recuerda el mismo patrón del módulo 1 con Word2Vec).
2. **$\sqrt{d_k}$**: factor de escala (donde $d_k$ es la dimensión de los vectores Key). Sin este escalamiento, los productos punto pueden crecer mucho cuando $d_k$ es grande, empujando al softmax hacia regiones donde el gradiente es casi cero. Dividir por $\sqrt{d_k}$ mantiene la varianza de las puntuaciones controlada.
3. **softmax**: convierte las puntuaciones en una distribución de probabilidad (suman 1) — son los **pesos de atención**.
4. **$\cdot V$**: promedio ponderado de los *values* usando esos pesos de atención. El resultado es una nueva representación de cada token, enriquecida con información del resto de la secuencia.

## 2.2 Multi-Head Attention

En lugar de calcular atención una sola vez, el Transformer la calcula varias veces en paralelo ("cabezas"), cada una con sus propias matrices $W^Q, W^K, W^V$ aprendidas independientemente:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

La intuición: cada cabeza puede aprender a fijarse en un tipo distinto de relación (una cabeza podría especializarse en relaciones sintácticas, otra en correferencia de pronombres, otra en proximidad posicional, etc.).

## 2.3 Positional Encoding

La atención por sí sola es **invariante al orden**: si permutas los tokens de entrada, los pesos de atención calculados sobre el mismo conjunto de tokens no cambian. Pero el orden de las palabras importa ("el perro mordió al cartero" ≠ "el cartero mordió al perro"). Para inyectar esta información, se suma a cada embedding una **codificación posicional**, frecuentemente basada en funciones seno y coseno de distintas frecuencias:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

(Los modelos modernos a menudo usan variantes como *rotary embeddings*, RoPE, pero la idea de fondo es la misma: codificar la posición de forma que el modelo pueda razonar sobre distancias relativas entre tokens.)

## 2.4 La arquitectura completa

Un bloque Transformer combina:

1. **Multi-Head Self-Attention** + conexión residual + normalización (`LayerNorm`)
2. **Feed-Forward Network** (una red densa simple aplicada a cada token de forma independiente) + conexión residual + normalización

```
Entrada
   │
   ├──► Multi-Head Attention ──► (+ residual) ──► LayerNorm
   │
   └──► Feed-Forward (MLP)   ──► (+ residual) ──► LayerNorm
   │
Salida
```

Un LLM moderno (como GPT) apila decenas de estos bloques (GPT-3 usa 96; modelos más recientes varían).

### Encoder vs. Decoder

El paper original propone una arquitectura **encoder-decoder** (útil para traducción). Los LLMs generativos modernos (familia GPT) usan **solo el decoder**, con una variante clave: **atención causal (masked)**, que impide que un token "vea" tokens futuros — esencial para que el modelo aprenda a predecir la siguiente palabra sin hacer trampa.

$$\text{(en la matriz de scores, se pone } -\infty \text{ en las posiciones futuras antes del softmax)}$$

## 2.5 ¿Cómo genera texto un LLM?

Una vez entrenado, un LLM generativo predice, token por token, una distribución de probabilidad sobre todo el vocabulario para la siguiente palabra, dado el contexto:

$$P(w_{t+1} \mid w_1, \dots, w_t)$$

y luego se muestrea (o se elige el más probable, *greedy decoding*, o con técnicas como *top-k*/*nucleus sampling*) para decidir cuál palabra generar. El proceso se repite token por token — por eso los LLMs "escriben" de forma secuencial incluso si entienden el contexto en paralelo.

## Código

Ver [`code/attention_demo.py`](./code/attention_demo.py) para una implementación ilustrativa de *scaled dot-product attention* en PyTorch puro (sin usar `nn.MultiheadAttention`), pensada para que veas la fórmula traducida línea por línea a código.

## Recursos recomendados

- Vaswani et al. (2017), *Attention Is All You Need* (el paper original).
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) (visualizaciones excelentes).
- [The Annotated Transformer — Harvard NLP](http://nlp.seas.harvard.edu/annotated-transformer/) (implementación completa, comentada línea por línea).
