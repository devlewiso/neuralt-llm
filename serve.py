"""Servidor local: entrena la red en vivo y muestra la visualización en el navegador.

Uso: python serve.py [--host 127.0.0.1] [--puerto 8765]
Abrí http://127.0.0.1:8765, escribí la tarea y dale a "Entrenar".
Los eventos llegan por Server-Sent Events (/stream); el registro de la corrida queda en saved/runs/.
"""
import argparse, json, queue, threading, time
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


PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>NeuralT · en vivo</title>
<style>
body{margin:0;background:#06040e;color:#dfe6ff;font-family:system-ui,sans-serif;display:flex;flex-direction:column;align-items:center}
h1{font-size:22px;margin:18px 0 4px;color:#fff}
#sub{color:#7fe6ff;font-size:14px;margin-bottom:12px}
form{display:flex;gap:8px;width:min(900px,95vw)}
input{flex:1;padding:10px;border-radius:8px;border:1px solid #3a2c5c;background:#120c20;color:#fff}
button{padding:10px 18px;border-radius:8px;border:0;background:#00c8e0;color:#001;font-weight:600;cursor:pointer}
button:disabled{opacity:.4}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;width:min(1000px,95vw);margin-top:14px}
canvas{background:#0b0716;border-radius:10px;width:100%}
#red{grid-column:1/3;height:300px}
#mapa{height:360px}
#curva{height:360px}
#log{grid-column:1/3;background:#0b0716;border-radius:10px;padding:10px;font:13px monospace;height:140px;overflow:auto;white-space:pre-wrap}
.k{color:#ffd25a}
</style></head><body>
<h1>Entrenada con datos de un LLM</h1>
<div id="sub">en vivo · esperando una corrida</div>
<form id="f"><input id="desc" value="Un NPC decide si huir (1) o pelear (0) segun su vida y la de su rival, ambas entre 0 y 1. Huye si su vida es baja comparada con la del rival."><button id="btn">Entrenar</button></form>
<div class="grid">
<canvas id="red"></canvas>
<canvas id="mapa"></canvas>
<canvas id="curva"></canvas>
<div id="log"></div>
</div>
<script>
const $ = id => document.getElementById(id);
const log = t => { const l = $('log'); l.textContent += t + '\\n'; l.scrollTop = l.scrollHeight; };
let snap = null, curva = [], fin = null, inicioCorrida = 0, esperando = false;

function dibujarRed(t) {
  const c = $('red'), g = c.getContext('2d');
  if (c.width !== c.clientWidth * 2) { c.width = c.clientWidth * 2; c.height = c.clientHeight * 2; }
  g.setTransform(2, 0, 0, 2, 0, 0);
  const w = c.clientWidth, h = c.clientHeight;
  const capas = [2, 8, 1];
  const xs = capas.map((_, k) => 60 + k * (w - 120) / (capas.length - 1));
  const pos = capas.map((n, k) => Array.from({length: n}, (_, j) => [xs[k], n === 1 ? h / 2 : 30 + j * (h - 60) / (n - 1)]));
  g.clearRect(0, 0, w, h);
  for (let L = 0; L < 2; L++) for (let i = 0; i < capas[L]; i++) for (let j = 0; j < capas[L + 1]; j++) {
    const wt = snap ? snap.pesos[L][i][j] : 0.3 * Math.sin(i * 1.7 + j * 2.3 + L);
    const [x1, y1] = pos[L][i], [x2, y2] = pos[L + 1][j];
    const a = Math.min(1, 0.15 + Math.abs(wt) * 0.6);
    g.strokeStyle = wt < 0 ? `rgba(255,80,200,${a})` : `rgba(0,220,255,${a})`;
    g.lineWidth = 1 + Math.min(2, Math.abs(wt));
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
    const s = (t * 0.6 + (i * 0.37 + j * 0.13 + L * 0.5) % 1) % 1;
    g.fillStyle = 'rgba(255,255,255,0.9)';
    g.beginPath(); g.arc(x1 + (x2 - x1) * s, y1 + (y2 - y1) * s, 3, 0, 7); g.fill();
  }
  pos.forEach(capa => capa.forEach(([x, y]) => { g.fillStyle = '#7fe6ff'; g.beginPath(); g.arc(x, y, 11, 0, 7); g.fill(); }));
}

function colorMapa(v) {
  v = Math.min(1, Math.max(0, v));
  if (v < .5) { const k = v / .5; return [40 + 20 * k, 10 + 190 * k, 120 + 135 * k]; }
  const k = (v - .5) / .5; return [60 + 195 * k, 200 + 30 * k, 255 - 200 * k];
}
function dibujarMapa() {
  const c = $('mapa'), g = c.getContext('2d');
  c.width = c.clientWidth * 2; c.height = c.clientHeight * 2; g.scale(2, 2);
  g.clearRect(0, 0, c.clientWidth, c.clientHeight);
  if (!snap) return;
  const m = snap.mapa, n = m.length, s = Math.min(c.clientWidth, c.clientHeight) - 40, ox = (c.clientWidth - s) / 2, oy = 30;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const [r, gg, b] = colorMapa(m[n - 1 - j][i]);
    g.fillStyle = `rgb(${r},${gg},${b})`;
    g.fillRect(ox + i * s / n, oy + j * s / n, s / n + 1, s / n + 1);
  }
  g.fillStyle = '#cfd8ff'; g.font = '13px sans-serif'; g.fillText('entrada 2 ↑', ox, oy - 8); g.fillText('entrada 1 →', ox + s - 90, oy + s + 16);
  g.fillText('mapa de respuesta de la red', ox, oy + s + 34);
}
function dibujarCurva() {
  const c = $('curva'), g = c.getContext('2d');
  c.width = c.clientWidth * 2; c.height = c.clientHeight * 2; g.scale(2, 2);
  const w = c.clientWidth, h = c.clientHeight;
  g.clearRect(0, 0, w, h);
  if (curva.length < 2) return;
  const mx = Math.max(...curva.map(p => p.e)), mn = Math.min(...curva.map(p => p.m)), mxm = Math.max(...curva.map(p => p.m));
  const ep = curva[curva.length - 1].e;
  g.strokeStyle = '#00e5ff'; g.lineWidth = 2.5; g.beginPath();
  curva.forEach((p, i) => {
    const x = 40 + (w - 60) * p.e / Math.max(1, ep), y = 20 + (h - 60) * (1 - (p.m - mn) / (mxm - mn + 1e-9));
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
  g.fillStyle = '#cfd8ff'; g.font = '13px sans-serif';
  g.fillText('error (MSE) · época ' + ep + ' · ' + curva[curva.length - 1].m.toFixed(4), 40, h - 12);
}

function pintar() { dibujarMapa(); dibujarCurva(); }
function bucle(ms) { dibujarRed(ms / 1000); requestAnimationFrame(bucle); }
requestAnimationFrame(bucle);
setInterval(() => {
  if (esperando && inicioCorrida) $('sub').textContent = 'pidiendo ejemplos al LLM… ' + Math.round((Date.now() - inicioCorrida) / 1000) + ' s';
}, 1000);

const es = new EventSource('/stream');
es.onmessage = ev => {
  const d = JSON.parse(ev.data);
  if (d.type === 'estado') {
    snap = d; esperando = false; curva.push({e: d.epoca, m: d.mse});
    $('sub').textContent = 'época ' + d.epoca + ' · error ' + d.mse.toFixed(4);
    pintar();
  } else if (d.type === 'llm_llamada') {
    log(`LLM ${d.modelo}: ${d.latencia_s} s · recibidos ${d.recibidos} · válidos ${d.validos} · descartados ${d.descartados}${d.error ? ' · ' + d.error : ''}`);
  } else if (d.type === 'log') {
    log(d.texto);
  } else if (d.type === 'fin') {
    $('sub').textContent = 'terminado · ' + d.epocas + ' épocas · error final ' + d.mse_final.toFixed(4);
    log('listo · corrida ' + d.corrida); $('btn').disabled = false;
  } else if (d.type === 'error') {
    log('ERROR: ' + d.mensaje); $('btn').disabled = false;
  } else if (d.type === 'tarea') {
    curva = []; snap = null; inicioCorrida = Date.now(); esperando = true; pintar(); $('log').textContent = '';
    $('btn').disabled = true;
  }
};
$('f').onsubmit = async e => {
  e.preventDefault();
  const r = await fetch('/iniciar', {method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({descripcion: $('desc').value})});
  const j = await r.json();
  if (!r.ok) { log('No se pudo iniciar: ' + j.error); }
};
pintar();
</script></body></html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--puerto", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.puerto), Handler)
    print(f"Servidor en http://{a.host}:{a.puerto}  (Ctrl+C para cerrar)")
    srv.serve_forever()
