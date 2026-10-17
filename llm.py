"""Puente con un LLM (API de Anthropic) para generar datos de entrenamiento.

Usa la variable de entorno ANTHROPIC_API_KEY. La clave nunca va en el código ni en el repo.
"""
import json
import os
import re
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
MODELO = os.environ.get("NEURALT_MODELO", "claude-haiku-5-5")


def preguntar(prompt, max_tokens=2000):
    clave = os.environ.get("ANTHROPIC_API_KEY")
    if not clave:
        raise RuntimeError("Falta ANTHROPIC_API_KEY en el entorno.")
    cuerpo = json.dumps({
        "model": MODELO,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=cuerpo, headers={
        "x-api-key": clave,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        respuesta = json.load(r)
    return "".join(b.get("text", "") for b in respuesta["content"])


def dataset_desde_llm(descripcion, entradas, salidas, n=40):
    """Pide al LLM n ejemplos (entrada -> salida) para la tarea descrita.

    Devuelve (X, Y) listas de floats. Valida que cada ejemplo tenga el tamaño correcto.
    """
    prompt = (
        f"{descripcion}\n\n"
        f"Genera exactamente {n} ejemplos. Cada entrada tiene {entradas} números entre 0 y 1; "
        f"cada salida tiene {salidas} números entre 0 y 1.\n"
        "Responde SOLO con un JSON así, sin texto extra:\n"
        '{"ejemplos": [{"in": [..], "out": [..]}, ...]}'
    )
    texto = preguntar(prompt, max_tokens=4000)
    bloque = re.search(r"\{.*\}", texto, re.S)
    datos = json.loads(bloque.group(0))
    X, Y = [], []
    for ej in datos["ejemplos"]:
        if len(ej["in"]) == entradas and len(ej["out"]) == salidas:
            X.append([float(v) for v in ej["in"]])
            Y.append([float(v) for v in ej["out"]])
    if not X:
        raise ValueError("El LLM no devolvió ejemplos válidos.")
    return X, Y
