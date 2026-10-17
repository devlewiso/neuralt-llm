"""Registro de eventos de cada corrida (un archivo JSONL por corrida) y reporte visual."""
import json
import os
import time
import uuid

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved", "runs")


class Corrida:
    def __init__(self, nombre):
        os.makedirs(CARPETA, exist_ok=True)
        self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{nombre}-{uuid.uuid4().hex[:6]}"
        self.ruta = os.path.join(CARPETA, self.id + ".jsonl")
        self.inicio = time.time()
        self.evento("inicio", nombre=nombre)

    def evento(self, tipo, **datos):
        registro = {"t": round(time.time() - self.inicio, 3), "tipo": tipo, **datos}
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return registro


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def reporte_png(ruta, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eventos = leer(ruta)
    epocas = [e for e in eventos if e["tipo"] == "epoca"]
    llamadas = [e for e in eventos if e["tipo"] == "llm_llamada"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0b0814")
    for ax in (ax1, ax2):
        ax.set_facecolor("#120c20")
        ax.tick_params(colors="#cfd8ff")
        for s in ax.spines.values():
            s.set_color("#3a2c5c")
    if epocas:
        ax1.plot([e["n"] for e in epocas], [e["mse"] for e in epocas], color="#00e5ff", lw=2)
        ax1.set_title("Error (MSE) durante el entrenamiento", color="white")
        ax1.set_xlabel("época", color="#cfd8ff")
        ax1.set_yscale("log")
    texto = ["Llamadas al LLM"]
    for e in llamadas:
        texto.append(
            f"- {e['modelo']}: {e['latencia_s']:.1f} s · recibidos {e['recibidos']} · "
            f"válidos {e['validos']} · descartados {e['descartados']} · {e.get('error', 'ok')}"
        )
    if not llamadas:
        texto.append("(sin llamadas)")
    ax2.axis("off")
    ax2.text(0.02, 0.9, "\n".join(texto), color="#cfd8ff", va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(salida, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return salida
