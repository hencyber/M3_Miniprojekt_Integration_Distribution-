FROM python:3.11-slim

WORKDIR /app

# installera uv
RUN pip install uv

# kopiera projektfiler
COPY pyproject.toml .
COPY app.py .
COPY model.onnx .

# installera dependencies med uv
RUN uv venv && uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
