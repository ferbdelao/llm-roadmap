"""
Fragmento ilustrativo de una API mínima con FastAPI que simula un pipeline
de pregunta-respuesta (con un paso opcional de RAG).

Para correrla de verdad:
    pip install fastapi uvicorn pydantic
    uvicorn api_demo:app --reload

Luego visita http://127.0.0.1:8000/docs para ver la documentación
interactiva generada automáticamente por FastAPI.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM Roadmap - Demo API")


class PreguntaRequest(BaseModel):
    pregunta: str
    usar_rag: bool = True


class RespuestaResponse(BaseModel):
    respuesta: str
    documentos_usados: list[str] = []


# --- Simulación de un pipeline de RAG (ver módulo 4 para la versión real) ---
def recuperar_documentos_relevantes(pregunta: str) -> list[str]:
    base_de_conocimiento = {
        "transformer": "El Transformer usa mecanismos de atención para procesar secuencias en paralelo.",
        "lora": "LoRA afina modelos grandes entrenando solo matrices de bajo rango.",
        "rag": "RAG combina recuperación de documentos con generación de texto.",
    }
    pregunta_lower = pregunta.lower()
    return [doc for clave, doc in base_de_conocimiento.items() if clave in pregunta_lower]


# --- Simulación de la llamada al LLM (aquí iría tu modelo local o una API externa) ---
def generar_respuesta(pregunta: str, contexto: list[str]) -> str:
    if contexto:
        return f"(Respuesta simulada usando {len(contexto)} documento(s) de contexto) " \
               f"Basado en lo recuperado, la respuesta a '{pregunta}' se relaciona con: {contexto[0]}"
    return f"(Respuesta simulada sin contexto adicional) No encontré información específica sobre: {pregunta}"


@app.post("/preguntar", response_model=RespuestaResponse)
def preguntar(request: PreguntaRequest):
    documentos = []
    if request.usar_rag:
        documentos = recuperar_documentos_relevantes(request.pregunta)

    respuesta = generar_respuesta(request.pregunta, documentos)

    return RespuestaResponse(respuesta=respuesta, documentos_usados=documentos)


@app.get("/")
def root():
    return {"status": "ok", "mensaje": "API de demostración del LLM Roadmap"}


# --------------------------------------------------------------------------
# Esqueleto de cómo se vería reemplazando la simulación por un LLM real
# vía API externa (ejemplo conceptual, no ejecutable sin tu propia API key):
# --------------------------------------------------------------------------
#
# import os
# import anthropic
#
# client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
#
# def generar_respuesta_real(pregunta: str, contexto: list[str]) -> str:
#     prompt = f"Contexto:\n{chr(10).join(contexto)}\n\nPregunta: {pregunta}"
#     mensaje = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=500,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return mensaje.content[0].text
