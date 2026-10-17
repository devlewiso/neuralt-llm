"""Red neuronal MLP minima en numpy: forward, backprop, guardar y cargar en JSON."""
import json
import numpy as np

ACTIVACIONES = {
    "linear": (lambda x: x, lambda y: np.ones_like(y)),
    "relu": (lambda x: np.maximum(0, x), lambda y: (y > 0).astype(float)),
    "sigmoid": (lambda x: 1 / (1 + np.exp(-x)), lambda y: y * (1 - y)),
    "tanh": (np.tanh, lambda y: 1 - y ** 2),
}


class Red:
    def __init__(self, capas, activaciones=None, semilla=0):
        """capas: lista de tamaños, p. ej. [2, 4, 1]."""
        rng = np.random.default_rng(semilla)
        self.capas = capas
        self.activaciones = activaciones or (["relu"] * (len(capas) - 2) + ["sigmoid"])
        self.pesos = [rng.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(capas[:-1], capas[1:])]
        self.sesgos = [np.zeros(b) for b in capas[1:]]

    def forward(self, x):
        """Devuelve la lista de activaciones de cada capa (incluida la entrada)."""
        salidas = [np.asarray(x, dtype=float)]
        for w, b, act in zip(self.pesos, self.sesgos, self.activaciones):
            salidas.append(ACTIVACIONES[act][0](salidas[-1] @ w + b))
        return salidas

    def predecir(self, x):
        return self.forward(x)[-1]

    def entrenar(self, X, Y, epocas=2000, lr=0.05, tol=None):
        """Descenso de gradiente por lotes con backpropagation. Devuelve el historial de MSE."""
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        historial = []
        for _ in range(epocas):
            acts = self.forward_lote(X)
            err = acts[-1] - Y
            mse = float(np.mean(err ** 2))
            historial.append(mse)
            if tol is not None and mse <= tol:
                break
            delta = err * ACTIVACIONES[self.activaciones[-1]][1](acts[-1])
            for i in reversed(range(len(self.pesos))):
                grad_w = acts[i].T @ delta / len(X)
                grad_b = delta.mean(axis=0)
                if i > 0:
                    delta = (delta @ self.pesos[i].T) * ACTIVACIONES[self.activaciones[i - 1]][1](acts[i])
                self.pesos[i] -= lr * grad_w
                self.sesgos[i] -= lr * grad_b
        return historial

    def forward_lote(self, X):
        salidas = [X]
        for w, b, act in zip(self.pesos, self.sesgos, self.activaciones):
            salidas.append(ACTIVACIONES[act][0](salidas[-1] @ w + b))
        return salidas

    def guardar(self, ruta):
        datos = {
            "version": "neuralt-llm-1",
            "capas": self.capas,
            "activaciones": self.activaciones,
            "pesos": [w.tolist() for w in self.pesos],
            "sesgos": [b.tolist() for b in self.sesgos],
        }
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        red = cls(datos["capas"], datos["activaciones"])
        red.pesos = [np.array(w) for w in datos["pesos"]]
        red.sesgos = [np.array(b) for b in datos["sesgos"]]
        return red
