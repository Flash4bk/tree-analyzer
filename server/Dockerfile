FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    curl ca-certificates libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Скачиваем модели из GitHub Releases ---
RUN mkdir -p models \
 && curl -L "https://github.com/Flash4bk/tree-analyzer/releases/download/v1-models/tree_model.pt" -o models/tree_model.pt \
 && curl -L "https://github.com/Flash4bk/tree-analyzer/releases/download/v1-models/stick_model.pt" -o models/stick_model.pt \
 && curl -L "https://github.com/Flash4bk/tree-analyzer/releases/download/v1-models/classifier.pth" -o models/classifier.pth

# --- Копируем код ---
COPY server .

ENV TREE_MODEL_PATH=models/tree_model.pt
ENV STICK_MODEL_PATH=models/stick_model.pt
ENV CLASSIFIER_PATH=models/classifier.pth
ENV REAL_STICK_LENGTH_M=1.0

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
