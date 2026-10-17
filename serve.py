"""Servidor local: entrena la red en vivo y muestra la visualización en el navegador.

Uso: python serve.py [--host 127.0.0.1] [--puerto 8765]
Abrí http://127.0.0.1:8765, escribí la tarea y dale a "Entrenar".
Los eventos llegan por Server-Sent Events (/stream); el registro de la corrida queda en saved/runs/.
"""
import argparse, json, os, queue, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
from nn import Red
from observe import Corrida
from llm import dataset_desde_llm

clientes = set()
clientes_lock = threading.Lock()
estado = {"ocupado": False}
ultimo = {}  # último snapshot, para clientes que se conectan a mitad de corrida


def difundir(evento):
    ultimo["snap"] = evento if evento["type"] == "estado" else ultimo.get("snap")
    with clientes_lock:
        for q in list(clientes):
            q.put(evento)


def mapa_respuesta(red, n=60):
    g = np.linspace(0, 1, n)
    X = np.array([[a, b] for b in g for a in g])
    return red.forward_lote(X)[-1][:, 0].reshape(n, n).round(3).tolist()


def entrenar_en_vivo(descripcion):
    corrida = Corrida("vivo")
    def observar(tipo, **datos):
        corrida.evento(tipo, **datos)
        difundir({"type": tipo, **datos})
    try:
        observar("tarea", descripcion=descripcion)
        difundir({"type": "log", "texto": "Pidiendo ejemplos al LLM…"})
        X, Y = dataset_desde_llm(descripcion, entradas=2, salidas=1, observar=observar)
        red = Red([2, 8, 1], semilla=1)
        difundir({"type": "log", "texto": f"{len(X)} ejemplos válidos. Entrenando…"})

        def al_avanzar(n, mse):
            if n % 10 == 0 or n == 1:
                difundir({"type": "estado", "epoca": n, "mse": mse,
                          "pesos": [w.round(4).tolist() for w in red.pesos],
                          "sesgos": [b.round(4).tolist() for b in red.sesgos],
                          "mapa": mapa_respuesta(red)})
                corrida.evento("epoca", n=n, mse=mse)

        historial = red.entrenar(X, Y, epocas=5000, lr=0.5, tol=0.001, al_avanzar=al_avanzar)
        red.guardar("saved/vivo.json")
        corrida.evento("fin", epocas=len(historial), mse_final=historial[-1])
        difundir({"type": "fin", "epocas": len(historial), "mse_final": historial[-1],
                  "corrida": corrida.id})
    except Exception as e:
        observar("error", mensaje=f"{type(e).__name__}: {e}")
        difundir({"type": "error", "mensaje": f"{type(e).__name__}: {e}"})
    finally:
        estado["ocupado"] = False




class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            cuerpo = PAGINA.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.end_headers()
            q = queue.Queue()
            with clientes_lock:
                clientes.add(q)
            try:
                if ultimo.get("snap"):
                    self.wfile.write(f"data: {json.dumps(ultimo['snap'])}\n\n".encode())
                    self.wfile.flush()
                while True:
                    try:
                        ev = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with clientes_lock:
                    clientes.discard(q)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/iniciar":
            return self.send_error(404)
        largo = int(self.headers.get("content-length", 0))
        datos = json.loads(self.rfile.read(largo) or b"{}")
        descripcion = (datos.get("descripcion") or "").strip()
        if not descripcion:
            return self._json(400, {"error": "Falta la descripción de la tarea."})
        if estado["ocupado"]:
            return self._json(409, {"error": "Ya hay una corrida en curso."})
        estado["ocupado"] = True
        ultimo.pop("snap", None)
        threading.Thread(target=entrenar_en_vivo, args=(descripcion,), daemon=True).start()
        self._json(200, {"ok": True})

    def _json(self, codigo, cuerpo):
        b = json.dumps(cuerpo).encode("utf-8")
        self.send_response(codigo)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


PAGINA = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pagina.html"), encoding="utf-8").read()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--puerto", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.puerto), Handler)
    print(f"Servidor en http://{a.host}:{a.puerto}  (Ctrl+C para cerrar)")
    srv.serve_forever()
