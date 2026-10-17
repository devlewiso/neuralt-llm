"""Video vertical 9:16 de una red entrenada: la red con sus pesos, el mapa de respuesta
a todas las entradas posibles y la curva de error reproducida desde el registro de la corrida.

Uso: python render_red.py saved/llm.json saved/runs/<corrida>.jsonl salida.mp4 [segundos]
"""
import math, os, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nn import Red
from observe import leer

RED_RUTA, CORRIDA, SALIDA = sys.argv[1], sys.argv[2], sys.argv[3]
SEG = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
W, H, FPS = 1080, 1920, 30
F_TIT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
F_TXT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
F_CHI = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

red = Red.cargar(RED_RUTA)
capas = red.capas
epocas = [e for e in leer(CORRIDA) if e["tipo"] == "epoca"]
fin = [e for e in leer(CORRIDA) if e["tipo"] == "fin"]
mse_final = fin[0]["mse_final"] if fin else None

# Mapa de respuesta: salida para cada combinación de (x1, x2) en [0,1]².
N = 90
malla = np.linspace(0, 1, N)
mapa = np.zeros((N, N))
for i, a in enumerate(malla):
    for j, b in enumerate(malla):
        mapa[j, i] = red.predecir([a, b])[0]

def color_mapa(v):
    # de violeta (0) a cian (0.5) a amarillo (1)
    v = min(1, max(0, v))
    if v < 0.5:
        k = v / 0.5
        return (int(40 + 20 * k), int(10 + 190 * k), int(120 + 135 * k))
    k = (v - 0.5) / 0.5
    return (int(60 + 195 * k), int(200 + 30 * k), int(255 - 200 * k))

# Posiciones de la red (entrada a la izquierda, salida a la derecha).
pos = []
xs = np.linspace(170, 910, len(capas))
for k, n in enumerate(capas):
    ys = np.linspace(470, 880, n) if n > 1 else [675]
    pos.append([(xs[k], y) for y in ys])

def activaciones(entrada):
    return red.forward(entrada)

fondo = Image.new("RGB", (W, H), (6, 4, 14))
total = int(SEG * FPS)
AQUI = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(AQUI, "saved"), exist_ok=True)
tmp = tempfile.mkdtemp(prefix="redviz-", dir=os.path.join(AQUI, "saved"))
ruta_sweep = [(0.5 + 0.42 * math.cos(2 * math.pi * k / 240), 0.5 + 0.42 * math.sin(2 * math.pi * k / 240)) for k in range(240)]

for fr in range(total):
    t = fr / FPS
    img = fondo.copy()
    d = ImageDraw.Draw(img, "RGBA")
    k = int(fr / total * len(ruta_sweep)) % len(ruta_sweep)
    entrada = list(ruta_sweep[k])
    acts = activaciones(entrada)
    salida = float(acts[-1][0])

    d.text((W / 2, 110), "Entrenada con datos de un LLM", font=F_TIT, fill=(255, 255, 255, 255), anchor="mm")
    d.text((W / 2, 190), "pesos reales de la corrida", font=F_CHI, fill=(120, 230, 255, 230), anchor="mm")

    # Red: conexiones y señales
    for c in range(len(capas) - 1):
        W_c = red.pesos[c]
        for i, (x1, y1) in enumerate(pos[c]):
            for j, (x2, y2) in enumerate(pos[c + 1]):
                w = float(W_c[i, j])
                alfa = min(1, 0.15 + abs(w) * 0.6)
                col = (255, 80, 200) if w < 0 else (0, 220, 255)
                d.line([(x1, y1), (x2, y2)], fill=col + (int(255 * alfa * 0.5),), width=2)
                s = (t * 0.8 + (i * 0.37 + j * 0.13) % 1) % 1
                px, py = x1 + (x2 - x1) * s, y1 + (y2 - y1) * s
                d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255, int(200 * alfa)))
    for c, capa in enumerate(pos):
        for i, (x, y) in enumerate(capa):
            v = min(1, max(0, float(acts[c][i]) if c > 0 else entrada[i]))
            r = 14 + 10 * v
            d.ellipse([x - r * 1.8, y - r * 1.8, x + r * 1.8, y + r * 1.8], fill=(0, 230, 255, int(50 * v)))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(int(120 + 135 * v), int(220 + 35 * v), 255, 255))

    # Mapa de respuesta
    x0, y0, lado = 190, 940, 700
    for j in range(N):
        for i in range(N):
            c = color_mapa(mapa[j, i])
            px0 = x0 + i * lado / N
            py0 = y0 + (N - 1 - j) * lado / N
            d.rectangle([px0, py0, px0 + lado / N + 1, py0 + lado / N + 1], fill=c + (255,))
    mx = x0 + entrada[0] * lado
    my = y0 + (1 - entrada[1]) * lado
    d.ellipse([mx - 16, my - 16, mx + 16, my + 16], outline=(255, 255, 255, 255), width=4)
    d.text((x0, y0 - 30), "↑ entrada 2", font=F_CHI, fill=(200, 200, 230, 255))
    d.text((x0, y0 + lado + 14), "0", font=F_CHI, fill=(200, 200, 230, 255))
    d.text((x0 + lado, y0 + lado + 14), "entrada 1 →", font=F_CHI, fill=(200, 200, 230, 255), anchor="ra")
    d.text((W / 2, y0 + lado + 62), f"entrada ({entrada[0]:.2f}, {entrada[1]:.2f})  →  salida {salida:.2f}",
           font=F_TXT, fill=(255, 210, 90, 255), anchor="mm")

    # Curva de error, reproducida con el avance del video
    gx0, gy0, gw, gh = 160, 1800, 760, 90
    d.text((gx0, gy0 - 36), "Error (MSE) durante el entrenamiento", font=F_CHI, fill=(200, 200, 230, 255))
    if epocas:
        mx_ep = epocas[-1]["n"]
        m_max = max(e["mse"] for e in epocas)
        m_min = min(e["mse"] for e in epocas)
        frac = min(1, fr / (total * 0.85))
        hasta = [e for e in epocas if e["n"] <= frac * mx_ep]
        pts = [(gx0 + gw * e["n"] / mx_ep, gy0 + gh * (1 - (e["mse"] - m_min) / (m_max - m_min + 1e-9))) for e in hasta]
        if len(pts) > 1:
            d.line(pts, fill=(0, 229, 255, 255), width=5)
        if mse_final is not None:
            d.text((gx0 + gw, gy0 - 36), f"final {mse_final:.3f}", font=F_CHI, fill=(0, 229, 255, 255), anchor="ra")
    img.save(os.path.join(tmp, f"f{fr:05d}.png"))

subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", SALIDA], check=True)
shutil.rmtree(tmp)
print("listo:", SALIDA)
