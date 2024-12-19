FROM python:3.9-slim

WORKDIR /app

RUN pip install fastapi uvicorn[standard] minio prometheus_client pyspark

ENV MINIO_ENDPOINT=minio:9000
ENV MINIO_ACCESS_KEY=minioadmin
ENV MINIO_SECRET_KEY=minioadmin
ENV BUCKET_NAME=movielens-bucket

COPY app.py /app/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
