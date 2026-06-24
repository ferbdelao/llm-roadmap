# 5. Despliegue Full Stack: Llevando el Modelo a Producción

Tener un modelo que funciona en un notebook es solo la mitad del trabajo. Este módulo cubre cómo exponer un LLM (o un pipeline RAG completo) como un servicio web real, accesible desde un navegador o una aplicación.

## 5.1 Arquitectura general de un sistema LLM en producción

```
┌─────────────┐      HTTP/JSON      ┌──────────────────┐      ┌────────────────┐
│  Frontend    │ ──────────────────► │   API (FastAPI)   │ ───► │  Modelo / LLM   │
│ (React/HTML) │ ◄────────────────── │  + lógica de RAG  │ ◄─── │ (local o vía API)│
└─────────────┘      Respuesta       └──────────────────┘      └────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ Base de datos     │
                                    │ vectorial (RAG)    │
                                    └──────────────────┘
```

## 5.2 ¿Modelo propio o API de terceros?

Antes de escribir código, hay una decisión de arquitectura importante:

| | Modelo propio (self-hosted) | API de un proveedor (OpenAI, Anthropic, etc.) |
|---|---|---|
| Costo inicial | Alto (GPU, infraestructura) | Bajo (pago por uso) |
| Control y privacidad de datos | Total | Depende del proveedor |
| Latencia / escalabilidad | Responsabilidad propia | Gestionada por el proveedor |
| Mantenimiento | Requiere DevOps/MLOps | Mínimo |

Muchos proyectos académicos y prototipos usan una API externa para la parte generativa, y reservan el cómputo propio para el pipeline de RAG (embeddings, base vectorial), que es mucho más ligero.

## 5.3 Construyendo la API con FastAPI

**FastAPI** es un framework de Python moderno, rápido y con documentación automática (Swagger UI), ideal para exponer un modelo como un servicio HTTP.

Los componentes típicos de un endpoint de inferencia:

1. Recibir la pregunta del usuario (request body, validado con un modelo de datos `Pydantic`).
2. Ejecutar el pipeline (RAG si aplica → construcción de prompt → llamada al modelo).
3. Devolver la respuesta como JSON.

Ver el fragmento completo en [`code/api_demo.py`](./code/api_demo.py).

## 5.4 Consideraciones clave para producción

### Streaming de respuestas

Esperar a que el modelo genere la respuesta completa antes de mostrar nada produce una mala experiencia de usuario (puede tardar varios segundos). La solución estándar es el **streaming**: enviar cada token generado al frontend en cuanto está disponible, usando *Server-Sent Events* (SSE) o WebSockets, para que el usuario vea el texto aparecer progresivamente (como en ChatGPT).

### Manejo de contexto y memoria de conversación

Un LLM no tiene memoria entre llamadas a la API por defecto: cada request es independiente. Para sostener una conversación, el backend debe reenviar el historial relevante en cada llamada (limitado por la ventana de contexto del modelo), o resumir conversaciones largas para no exceder ese límite.

### Costos y límites de tasa (rate limiting)

Cada llamada a un LLM (propio o de terceros) tiene un costo computacional o monetario. En producción es importante:
- Limitar la longitud de los prompts y respuestas.
- Implementar *rate limiting* por usuario.
- Cachear respuestas a preguntas frecuentes cuando sea razonable.

### Seguridad básica

- Nunca expongas claves de API en el frontend; deben vivir únicamente en el backend (variables de entorno).
- Valida y sanitiza la entrada del usuario antes de insertarla en un prompt (mitigar *prompt injection* básico).
- Considera límites de longitud de entrada para evitar abusos.

## 5.5 El frontend

No necesita ser sofisticado: un formulario simple en HTML/JavaScript (o React, si quieres algo más robusto) que:
1. Envía la pregunta del usuario al endpoint de la API.
2. Muestra la respuesta (idealmente, consumiendo el stream de tokens).

Ver un esqueleto mínimo en [`code/frontend_demo.html`](./code/frontend_demo.html).

## Código

- [`code/api_demo.py`](./code/api_demo.py): API mínima con FastAPI que recibe una pregunta, simula un pipeline de RAG y devuelve una respuesta.
- [`code/frontend_demo.html`](./code/frontend_demo.html): página HTML mínima que consume esa API.

## Recursos recomendados

- [Documentación oficial de FastAPI](https://fastapi.tiangolo.com/)
- [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints) (para desplegar modelos propios sin gestionar servidores desde cero)
- [Documentación de Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

## Cierre del roadmap

Si llegaste hasta aquí, recorriste el camino completo: desde representar una palabra como un vector ([módulo 1](../01-fundamentos/README.md)), entender cómo un Transformer relaciona esas palabras entre sí ([módulo 2](../02-transformer/README.md)), adaptar un modelo preentrenado a un dominio específico ([módulo 3](../03-fine-tuning/README.md)), dotarlo de conocimiento externo actualizado ([módulo 4](../04-rag/README.md)), hasta exponerlo como un servicio real al mundo. El siguiente paso natural es **construir algo propio**: elige un dataset que te interese y arma tu propia versión, aunque sea pequeña, de este pipeline completo.
