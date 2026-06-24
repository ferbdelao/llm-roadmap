# La Ruta de un LLM: De las Matemáticas al Despliegue

¡Hola! Este repositorio es una guía paso a paso creada para entender el ciclo de vida completo de un *Large Language Model* (LLM), pensada especialmente para estudiantes de **Matemáticas Aplicadas y Computación**, y entusiastas de los datos en general.

No necesitas saber nada de Deep Learning para empezar: cada módulo construye sobre el anterior, partiendo de álgebra lineal y probabilidad básicas hasta llegar a un modelo funcionando detrás de una API web.

## Objetivo

Desmitificar la "caja negra" de los LLMs. Aquí no solo consumimos APIs: explicamos las matemáticas detrás de cada componente, la arquitectura que las conecta, y mostramos fragmentos de código que ilustran cómo se traduce la teoría a la práctica.

## Filosofía de la guía

- **Nivel matemático intermedio**: se presentan y explican las fórmulas clave (con su intuición), pero no se derivan exhaustivamente todos los pasos. El objetivo es que entiendas *qué* hace cada ecuación y *por qué* está ahí, no memorizar una demostración completa.
- **Código ilustrativo, no de producción**: los fragmentos de código (Python / PyTorch) sirven para conectar la teoría con su implementación. No son notebooks listos para correr de punta a punta con un solo clic, sino piezas claras y comentadas que puedes adaptar y ejecutar tú mismo.
- **Una carpeta por tema**: cada etapa del roadmap vive en su propia carpeta, con su propio `README.md` explicativo y una subcarpeta `code/` con los fragmentos correspondientes.

## Mapa de Ruta (Roadmap)

| # | Módulo | ¿De qué trata? |
|---|--------|-----------------|
| 1 | [Fundamentos](./01-fundamentos/README.md) | Vectores, embeddings, probabilidad y NLP clásico: el lenguaje matemático que necesitas antes de tocar un Transformer. |
| 2 | [El Transformer](./02-transformer/README.md) | Mecanismos de atención, *self-attention*, *multi-head attention* y la arquitectura que cambió el NLP para siempre. |
| 3 | [Fine-Tuning](./03-fine-tuning/README.md) | Cómo adaptar un LLM preentrenado a tus propios datos: *full fine-tuning*, LoRA y *prompt tuning*. |
| 4 | [RAG](./04-rag/README.md) | *Retrieval-Augmented Generation*: dándole memoria externa y actualizada al modelo. |
| 5 | [Despliegue Full Stack](./05-despliegue/README.md) | Llevar el modelo a producción: API con FastAPI, frontend simple y consideraciones de servir un LLM en el mundo real. |

```
llm-roadmap/
├── 01-fundamentos/
│   ├── README.md
│   └── code/
├── 02-transformer/
│   ├── README.md
│   └── code/
├── 03-fine-tuning/
│   ├── README.md
│   └── code/
├── 04-rag/
│   ├── README.md
│   └── code/
└── 05-despliegue/
    ├── README.md
    └── code/
```

## ¿Cómo usar este repositorio?

1. Sigue los módulos en orden: cada uno asume conocimientos del anterior.
2. Lee primero el `README.md` de cada carpeta (teoría + intuición).
3. Revisa los fragmentos en `code/` para ver cómo se vería en la práctica.
4. No te quedes solo leyendo: instala las librerías mencionadas y experimenta modificando los fragmentos.

## Tecnologías utilizadas

- **Python** como lenguaje base
- **PyTorch** para los fragmentos de redes neuronales
- **Hugging Face Transformers** para cargar y afinar modelos preentrenados
- **LangChain / LlamaIndex** para los pipelines de RAG
- **FastAPI** + **React/HTML** para el despliegue web

## Prerrequisitos sugeridos

- Álgebra lineal: vectores, matrices, producto punto, normas.
- Probabilidad básica: distribuciones, esperanza, la función softmax.
- Cálculo: derivadas parciales y la idea de gradiente (no necesitas dominar backpropagation a fondo, solo la intuición).
- Programación en Python a nivel intermedio.

## Contribuciones

Este es un proyecto académico/personal en construcción. Si encuentras un error o quieres proponer una mejora, abre un *issue* o un *pull request*.

## Licencia

Este contenido se distribuye con fines educativos. Siéntete libre de usarlo y adaptarlo citando la fuente.
