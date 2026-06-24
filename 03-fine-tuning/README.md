# 3. Fine-Tuning: Adaptando un LLM a tus Datos

Un LLM preentrenado (como Llama, Mistral o GPT) ya "sabe" lenguaje en general, pero no necesariamente sabe responder en el tono, dominio o formato que tú necesitas. El **fine-tuning** (afinación) es el proceso de continuar entrenando un modelo preentrenado sobre un conjunto de datos más pequeño y específico.

## 3.1 ¿Por qué no entrenar desde cero?

Entrenar un LLM desde cero requiere:
- Cientos de miles de millones de tokens de texto.
- Cientos o miles de GPUs corriendo durante semanas.
- Presupuestos de millones de dólares.

El *transfer learning* (aprendizaje por transferencia) evita esto: aprovechamos que el modelo ya aprendió gramática, hechos generales y razonamiento básico durante el preentrenamiento, y solo ajustamos sus pesos para la tarea específica que nos interesa.

## 3.2 Full Fine-Tuning

La forma más directa: actualizar **todos** los parámetros del modelo usando tu propio dataset, con una función de pérdida estándar (típicamente *cross-entropy* sobre la predicción del siguiente token).

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t \mid w_1, \dots, w_{t-1})$$

**Problema práctico**: un LLM moderno tiene miles de millones de parámetros. Actualizarlos todos requiere guardar gradientes y estados del optimizador para cada uno, lo cual demanda una cantidad de memoria GPU que muy pocas organizaciones pueden permitirse para modelos grandes.

## 3.3 Parameter-Efficient Fine-Tuning (PEFT)

La alternativa moderna: en lugar de actualizar todos los pesos, congelamos el modelo preentrenado y entrenamos solo un pequeño número de parámetros adicionales.

### LoRA (Low-Rank Adaptation)

La idea de LoRA (Hu et al., 2021): en lugar de actualizar una matriz de pesos completa $W \in \mathbb{R}^{d \times k}$, se aproxima su actualización $\Delta W$ mediante el producto de dos matrices de **rango bajo**:

$$\Delta W = B A, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$

Durante el fine-tuning, $W$ permanece congelada y solo se entrenan $A$ y $B$:

$$W' = W + \Delta W = W + BA$$

**¿Por qué funciona?** La hipótesis (respaldada empíricamente) es que la actualización necesaria para adaptar un modelo a una tarea específica vive, en la práctica, en un subespacio de dimensión mucho menor que el espacio completo de parámetros. Con $r$ tan pequeño como 4 u 8, se pueden obtener resultados competitivos con *full fine-tuning*, entrenando una fracción mínima de los parámetros (a veces menos del 1%).

**Ventajas prácticas**:
- Mucho menos uso de memoria GPU (no hay que guardar gradientes de todo el modelo).
- Los pesos LoRA resultantes son pequeños (megabytes, no gigabytes) y se pueden intercambiar fácilmente para distintas tareas sobre el mismo modelo base.

### QLoRA

Una extensión de LoRA que además **cuantiza** el modelo base a 4 bits (en lugar de los 16 o 32 bits habituales) antes de aplicar los adaptadores LoRA, reduciendo aún más el uso de memoria y permitiendo afinar modelos grandes en una sola GPU de consumo.

## 3.4 Instruction Tuning y RLHF (panorama general)

Más allá de adaptar el modelo a un dominio específico, existe otro tipo de fine-tuning fundamental para los asistentes conversacionales modernos:

1. **Instruction Tuning (SFT, *Supervised Fine-Tuning*)**: se entrena el modelo con pares de (instrucción, respuesta deseada) para que aprenda a seguir instrucciones en lugar de simplemente completar texto.
2. **RLHF (*Reinforcement Learning from Human Feedback*)**: se entrena un modelo de recompensa a partir de comparaciones humanas entre respuestas, y luego se usa ese modelo de recompensa para ajustar el LLM mediante aprendizaje por refuerzo (típicamente con algoritmos como PPO), de forma que genere respuestas más alineadas con las preferencias humanas.

Este último tema merece su propio estudio profundo (es un curso en sí mismo); aquí solo lo mencionamos para que sepas dónde ubicarlo dentro del panorama general.

## 3.5 Prompt Tuning / Prefix Tuning (mención breve)

Otra familia de técnicas PEFT que, en lugar de modificar pesos del modelo, aprende un pequeño conjunto de "vectores de prompt" (*soft prompts*) que se concatenan a la entrada y se optimizan directamente — el modelo en sí permanece completamente congelado.

## Código

Ver [`code/lora_demo.py`](./code/lora_demo.py) para un fragmento ilustrativo de cómo se ve, conceptualmente, una capa lineal con un adaptador LoRA añadido, y cómo se configuraría usando la librería `peft` de Hugging Face.

## Recursos recomendados

- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*.
- Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*.
- [Documentación de la librería PEFT — Hugging Face](https://huggingface.co/docs/peft)
- Ouyang et al. (2022), *Training language models to follow instructions with human feedback* (paper de InstructGPT, base conceptual de RLHF en LLMs).
