# Lingkungan disamakan dengan makalah: Python 3.8, scikit-learn 1.3.2
FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY src/ src/

EXPOSE 8080
CMD ["python", "/app/src/serve.py"]
