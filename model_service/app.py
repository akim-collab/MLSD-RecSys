import os
import io
import tarfile
from fastapi import FastAPI, HTTPException
from minio import Minio
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

PREDICTION_COUNT = Counter('prediction_requests', 'Number of prediction requests')

app = FastAPI()

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "movielens-bucket")

MODEL_TAR = "als_model.tar.gz"
MODEL_DIR = "/app/als_model"

spark = None
model = None

def load_model_from_minio():
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    if not client.bucket_exists(BUCKET_NAME):
        raise Exception("Bucket does not exist.")
    response = client.get_object(BUCKET_NAME, MODEL_TAR)
    data = response.read()
    with open("/app/model.tar.gz", "wb") as f:
        f.write(data)
    with tarfile.open("/app/model.tar.gz", "r:gz") as tar:
        tar.extractall("/app")

@app.on_event("startup")
def startup_event():
    global spark, model
    load_model_from_minio()
    spark = SparkSession.builder \
        .appName("ModelService") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    model = ALSModel.load(MODEL_DIR)

@app.on_event("shutdown")
def shutdown_event():
    if spark:
        spark.stop()

@app.get("/predict")
def predict(user_id: int):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    PREDICTION_COUNT.inc()
    users = spark.createDataFrame([(user_id, )], ["userId"])
    userRecs = model.recommendForUserSubset(users, 5).collect()
    if len(userRecs) == 0:
        return {"user_id": user_id, "recommendations": []}
    recs = userRecs[0].recommendations
    return {
        "user_id": user_id,
        "recommendations": [{"movieId": r.movieId, "rating": r.rating} for r in recs]
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)