"""Puente con un LLM para generar datos de entrenamiento.

Backends (variable NEURALT_BACKEND):
  ollama     (por defecto) modelo local en el PC asgard, sin clave.
             OLLAMA_URL = http://192.168.1.100:11434 por defecto.
  anthropic  API de Anthropic. Necesita ANTHROPIC_API_KEY (nunca va en el código ni en el repo).
"""
import json
import os
import re
import urllib.request

BACKEND = os.environ.get("NEURALT_BACKEND", "ollama")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.1.100:11434")
API_URL = "https://api.anthropic.com/v1/messages"
MODELO = os.environ.get("NEURALT_MODELO") or ("llama3.1:8b" if BACKEND == "ollama" else "claude-haiku-5-5")


def preguntar_ollama(prompt, max_tokens=2000):
    cuerpo = json.dumps({
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=cuerpo,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["response"]


def preguntar(prompt, max_tokens=2000):
    if BACKEND == "ollama":
        return preguntar_ollama(prompt, max_tokens)
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


def dataset_desde_llm(descripcion, entradas, salidas, n=40, observar=None):
    """Pide al LLM n ejemplos (entrada -> salida) para la tarea descrita.

    Devuelve (X, Y) listas de floats. Valida que cada ejemplo tenga el tamaño correcto.
    observar(tipo, **datos) recibe las métricas de la llamada si se pasa.
    """
    import time
    inicio = time.time()
    registro = {"modelo": MODELO, "backend": BACKEND, "pedidos": n}
    prompt = (
        f"{descripcion}\n\n"
        f"Genera exactamente {n} ejemplos. Cada entrada tiene {entradas} números entre 0 y 1; "
        f"cada salida tiene {salidas} números entre 0 y 1.\n"
        "Responde SOLO con un JSON así, sin texto extra:\n"
        '{"ejemplos": [{"in": [..], "out": [..]}, ...]}'
    )
    error = None
    X, Y, recibidos = [], [], 0
    try:
        texto = preguntar(prompt, max_tokens=4000)
        bloque = re.search(r"\{.*\}", texto, re.S)
        datos = json.loads(bloque.group(0))
        recibidos = len(datos["ejemplos"])
        for ej in datos["ejemplos"]:
            if len(ej["in"]) == entradas and len(ej["out"]) == salidas:
                X.append([float(v) for v in ej["in"]])
                Y.append([float(v) for v in ej["out"]])
        if not X:
            error = "sin ejemplos válidos"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    registro.update({
        "latencia_s": round(time.time() - inicio, 2),
        "recibidos": recibidos,
        "validos": len(X),
        "descartados": recibidos - len(X),
        "error": error,
    })
    if observar:
        observar("llm_llamada", **registro)
    if error:
        raise ValueError(f"El LLM falló: {error}")
    return X, Y
