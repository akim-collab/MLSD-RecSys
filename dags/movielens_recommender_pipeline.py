import os
import logging
import requests
import zipfile
import pandas as pd
from io import BytesIO
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml.recommendation import ALS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
S3_BUCKET = "movielens-bucket"

LOCAL_DATA_DIR = "/shared_data"
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"
MODEL_DIR = "als_model"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
LOCAL_ZIP = "ml_latest_small.zip"
EXTRACT_DIR = "data"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

dag = DAG(
    'movielens_processing_pipeline',
    default_args=default_args,
    schedule_interval=None,
    description="Pipeline: Download, split, train model and store in Minio"
)

def download_dataset():
    logger.info("Downloading MovieLens dataset...")
    response = requests.get(MOVIELENS_URL)
    with open(LOCAL_ZIP, "wb") as f:
        f.write(response.content)
    logger.info("Dataset downloaded.")

def extract_dataset():
    if not os.path.exists(LOCAL_ZIP):
        logger.info("Archive not found.")
        return
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as z:
        z.extractall(EXTRACT_DIR)
    logger.info("Dataset extracted.")

def split_data():
    ratings_path = os.path.join(EXTRACT_DIR, "ml-latest-small", "ratings.csv")
    if not os.path.exists(ratings_path):
        logger.info("Ratings file not found.")
        return
    df = pd.read_csv(ratings_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    train = df[df['timestamp'] < '2015-01-01']
    test = df[df['timestamp'] >= '2015-01-01']
    train.to_csv(os.path.join(LOCAL_DATA_DIR, TRAIN_FILE), index=False)
    test.to_csv(os.path.join(LOCAL_DATA_DIR, TEST_FILE), index=False)
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")

def upload_to_minio():
    from minio import Minio
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    if not client.bucket_exists(S3_BUCKET):
        client.make_bucket(S3_BUCKET)
        logger.info(f"Bucket {S3_BUCKET} created in Minio.")
    else:
        logger.info("Bucket already exists.")

    train_path = os.path.join(LOCAL_DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(LOCAL_DATA_DIR, TEST_FILE)

    def upload_file(file_name):
        with open(os.path.join(LOCAL_DATA_DIR, file_name), 'rb') as f:
            data = f.read()
        client.put_object(S3_BUCKET, file_name, data=BytesIO(data), length=len(data), content_type='application/csv')

    upload_file(TRAIN_FILE)
    upload_file(TEST_FILE)
    logger.info("Data uploaded to Minio.")

def train_model():
    spark = (SparkSession.builder
             .appName("MovieLensModelTrain")
             .master("spark://spark-master:7077")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Config for Minio
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.access.key", MINIO_ACCESS_KEY)
    hadoop_conf.set("fs.s3a.secret.key", MINIO_SECRET_KEY)
    hadoop_conf.set("fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("com.amazonaws.services.s3.enableV4", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")

    schema = StructType([
        StructField("userId", IntegerType()),
        StructField("movieId", IntegerType()),
        StructField("rating", FloatType()),
        StructField("timestamp", IntegerType())
    ])

    train_df = spark.read.csv(f"s3a://{S3_BUCKET}/{TRAIN_FILE}", header=True, schema=schema)
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", maxIter=5, rank=10)
    model = als.fit(train_df)
    logger.info("Model trained.")

    local_model_path = os.path.join(LOCAL_DATA_DIR, MODEL_DIR)
    model.write().overwrite().save(local_model_path)
    logger.info("Model saved locally.")

    # Загрузим модель в Minio как tar-архив
    import tarfile
    tar_path = os.path.join(LOCAL_DATA_DIR, "als_model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_model_path, arcname=MODEL_DIR)

    from minio import Minio
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    with open(tar_path, 'rb') as f:
        data = f.read()
    client.put_object(S3_BUCKET, "als_model.tar.gz", data=BytesIO(data), length=len(data), content_type='application/gzip')
    logger.info("Model uploaded to Minio.")
    spark.stop()

download_task = PythonOperator(task_id='fetch_data', python_callable=download_dataset, dag=dag)
extract_task = PythonOperator(task_id='unzip_data', python_callable=extract_dataset, dag=dag)
split_task = PythonOperator(task_id='prepare_splits', python_callable=split_data, dag=dag)
upload_task = PythonOperator(task_id='push_to_minio', python_callable=upload_to_minio, dag=dag)
train_task = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)

download_task >> extract_task >> split_task >> upload_task >> train_task
