# neuralt-llm

Editor y entrenador de redes neuronales pequeñas, con un puente a un LLM para generar datos de entrenamiento.
Empezamos simple: una red MLP en numpy que se entrena, se guarda en JSON y se puede consultar.

## Uso

```bash
pip install numpy
python main.py xor                       # prueba: aprende XOR
python main.py predecir saved/xor.json 1 0
```

Para usar el LLM, exportá la clave en tu terminal (nunca en el código ni en el repo):

```bash
export ANTHROPIC_API_KEY=...             # tu clave
python main.py llm "Clasifica si un NPC debe huir (1) o pelear (0) según su vida y la de su rival, ambas 0–1"
```

Modelo por defecto: `claude-haiku-5-5`. Se cambia con `NEURALT_MODELO`.

## Estructura

- `nn.py`: red MLP (forward, backprop, guardar/cargar JSON).
- `llm.py`: llamada a la API de Anthropic y validación de los ejemplos que devuelve.
- `main.py`: comandos de línea.

## Próximos pasos

- Editor visual de capas (como el NeuralT original).
- Usar la red como "cerebro" de un NPC con varias acciones posibles.
