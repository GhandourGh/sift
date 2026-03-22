FROM python:3.11-slim

WORKDIR /app

COPY V3/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY V3/config.py .
COPY V3/pipeline.py .
COPY V3/web_app.py .
COPY V3/yolov8n.pt .
COPY V3/algorithms/ algorithms/
COPY V3/evaluation/ evaluation/
COPY V3/visualization/ visualization/
COPY V3/web/ web/

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
