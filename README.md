# M3 – Integration & Distribution

Klassificeringsmodell (CIFAR-10) från K2 som körs via ett FastAPI-API i en Docker-container.

## Hur man kör

### Lokalt (utan Docker)
```
uv sync
uv run python export_model.py
uv run uvicorn app:app --port 8000
```

### Med Docker
```
docker build -t m3-predict .
docker run -p 8000:8000 m3-predict
```

## Testa API:et

Skicka en POST till `/predict` med 3072 float-värden (3x32x32 bild):

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [0.1, -0.5, 0.3, ...]}'
```

Svar:
```
{"prediction": "truck", "class_id": 9}
```

## Filer

- `model.py` – CNN-modell från K2
- `export_model.py` – tränar och exporterar till ONNX
- `app.py` – FastAPI med POST /predict
- `Dockerfile` – kör API:et i container med uv

## Pull Requests

- [PR #1 – Modell-export till ONNX](https://github.com/hencyber/M3_Miniprojekt_Integration_Distribution-/pull/1)
- [PR #2 – FastAPI och Docker](https://github.com/hencyber/M3_Miniprojekt_Integration_Distribution-/pull/2)
