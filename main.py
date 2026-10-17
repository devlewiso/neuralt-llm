"""Uso:
  python main.py xor                 entrena una red con XOR y la guarda
  python main.py llm "descripcion"   pide datos al LLM, entrena y guarda
  python main.py predecir ruta.json 0.2 0.9
  python main.py reporte saved/runs/<corrida>.jsonl   genera el PNG con curvas y llamadas al LLM
"""
import os
import sys
from nn import Red
from observe import Corrida, reporte_png

XOR = ([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])


def entrenar_y_guardar(X, Y, ruta, corrida=None):
    red = Red([len(X[0]), 8, len(Y[0])], semilla=1)

    def al_avanzar(n, mse):
        if corrida and (n % 25 == 0 or n == 1):
            corrida.evento("epoca", n=n, mse=mse)

    historial = red.entrenar(X, Y, epocas=5000, lr=0.5, tol=0.001, al_avanzar=al_avanzar)
    red.guardar(ruta)
    if corrida:
        corrida.evento("fin", epocas=len(historial), mse_final=historial[-1], ruta=ruta)
    print(f"épocas: {len(historial)}  MSE final: {historial[-1]:.5f}  guardada en {ruta}")
    for x, y in zip(X, Y):
        print(x, "->", round(float(red.predecir(x)[0]), 3), "esperado", y[0])
    return red


if __name__ == "__main__":
    os.makedirs("saved", exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "xor"
    if cmd == "xor":
        entrenar_y_guardar(*XOR, "saved/xor.json")
    elif cmd == "llm":
        from llm import dataset_desde_llm
        descripcion = sys.argv[2]
        corrida = Corrida("llm")
        corrida.evento("tarea", descripcion=descripcion)
        X, Y = dataset_desde_llm(descripcion, entradas=2, salidas=1, observar=corrida.evento)
        entrenar_y_guardar(X, Y, "saved/llm.json", corrida)
        png = reporte_png(corrida.ruta, corrida.ruta.replace(".jsonl", ".png"))
        print("reporte:", png)
    elif cmd == "reporte":
        print(reporte_png(sys.argv[2], sys.argv[2].replace(".jsonl", ".png")))
    elif cmd == "predecir":
        red = Red.cargar(sys.argv[2])
        x = [float(v) for v in sys.argv[3:]]
        print(red.predecir(x))
    else:
        print(__doc__)
