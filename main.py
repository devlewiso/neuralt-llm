"""Uso:
  python main.py xor                 entrena una red con XOR y la guarda
  python main.py llm "descripcion"   pide datos al LLM, entrena y guarda
  python main.py predecir ruta.json 0.2 0.9
"""
import sys
from nn import Red

XOR = ([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])


def entrenar_y_guardar(X, Y, ruta):
    red = Red([len(X[0]), 8, len(Y[0])], semilla=1)
    historial = red.entrenar(X, Y, epocas=5000, lr=0.5, tol=0.001)
    red.guardar(ruta)
    print(f"épocas: {len(historial)}  MSE final: {historial[-1]:.5f}  guardada en {ruta}")
    for x, y in zip(X, Y):
        print(x, "->", round(float(red.predecir(x)[0]), 3), "esperado", y[0])
    return red


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "xor"
    if cmd == "xor":
        entrenar_y_guardar(*XOR, "saved/xor.json")
    elif cmd == "llm":
        from llm import dataset_desde_llm
        descripcion = sys.argv[2]
        X, Y = dataset_desde_llm(descripcion, entradas=2, salidas=1)
        entrenar_y_guardar(X, Y, "saved/llm.json")
    elif cmd == "predecir":
        red = Red.cargar(sys.argv[2])
        x = [float(v) for v in sys.argv[3:]]
        print(red.predecir(x))
    else:
        print(__doc__)
