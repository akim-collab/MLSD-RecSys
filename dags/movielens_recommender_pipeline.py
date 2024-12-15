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
from minio import Minio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация Minio и путей
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
S3_BUCKET = "movielens-bucket"

LOCAL_DATA_DIR = "/shared_data"
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"
MODEL_DIR = "als_saved_model"

# URL для скачивания датасета
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
    description="End-to-end pipeline для обучения рекомендаций MovieLens"
)

def download_dataset():
    """Скачивает архив с данными MovieLens."""
    logger.info("Начинаю загрузку архива с датасетом...")
    response = requests.get(MOVIELENS_URL)
    with open(LOCAL_ZIP, "wb") as f:
        f.write(response.content)
    logger.info("Архив загружен локально.")

def extract_dataset():
    """Распаковывает датасет в локальную директорию."""
    if not os.path.exists(LOCAL_ZIP):
        logger.info("Архив не найден, повторите загрузку.")
        return
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as z:
        z.extractall(EXTRACT_DIR)
    logger.info("Датасет успешно распакован.")

def split_data():
    """Делит данные на тренировочные и тестовые по дате."""
    ratings_path = os.path.join(EXTRACT_DIR, "ml-latest-small", "ratings.csv")
    if not os.path.exists(ratings_path):
        logger.info("Файл с рейтингами не найден.")
        return
    df = pd.read_csv(ratings_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    train = df[df['timestamp'] < '2015-01-01']
    test = df[df['timestamp'] >= '2015-01-01']
    train.to_csv(os.path.join(LOCAL_DATA_DIR, TRAIN_FILE), index=False)
    test.to_csv(os.path.join(LOCAL_DATA_DIR, TEST_FILE), index=False)
    logger.info(f"Тренировочный набор: {len(train)}, тестовый набор: {len(test)}")

def upload_to_minio():
    """Загружает тренировочные и тестовые данные в Minio."""
    client = Minio(MINIO_ENDPOINT,
                   access_key=MINIO_ACCESS_KEY,
                   secret_key=MINIO_SECRET_KEY,
                   secure=False)
    if not client.bucket_exists(S3_BUCKET):
        client.make_bucket(S3_BUCKET)
        logger.info(f"Создан бакет {S3_BUCKET} в Minio.")
    else:
        logger.info(f"Бакет {S3_BUCKET} уже существует.")

    train_path = os.path.join(LOCAL_DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(LOCAL_DATA_DIR, TEST_FILE)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_bytes = train_df.to_csv(index=False).encode('utf-8')
    test_bytes = test_df.to_csv(index=False).encode('utf-8')

    client.put_object(S3_BUCKET, TRAIN_FILE, data=BytesIO(train_bytes), length=len(train_bytes), content_type='application/csv')
    client.put_object(S3_BUCKET, TEST_FILE, data=BytesIO(test_bytes), length=len(test_bytes), content_type='application/csv')
    logger.info("Данные загружены в бакет Minio.")

def train_and_predict():
    """Обучает модель ALS на Spark и сохраняет предикты в Minio."""
    logger.info("Инициализация Spark сессии...")
    spark = (SparkSession.builder
             .appName("MovieLensALSRecommendation")
             .master("spark://spark-master:7077")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Конфиг для Minio
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

    logger.info("Чтение тренировочных и тестовых данных из Minio...")
    train_df = spark.read.csv(f"s3a://{S3_BUCKET}/{TRAIN_FILE}", header=True, schema=schema)
    test_df = spark.read.csv(f"s3a://{S3_BUCKET}/{TEST_FILE}", header=True, schema=schema)

    logger.info("Обучение модели ALS...")
    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", maxIter=5, rank=10)
    model = als.fit(train_df)

    # Сохраняем модель локально
    model_save_path = os.path.join(LOCAL_DATA_DIR, MODEL_DIR)
    model.write().overwrite().save(model_save_path)
    logger.info("Модель сохранена локально.")

    predictions = model.transform(test_df)
    logger.info("Предсказания готовы, сохраняю их в Minio...")

    # Сохраняем предикты в Minio в формате CSV
    predictions.select("userId","movieId","prediction") \
        .write.csv(f"s3a://{S3_BUCKET}/predictions_output", mode="overwrite", header=True)
    logger.info("Предсказания сохранены в Minio.")

    spark.stop()


download_task = PythonOperator(
    task_id='fetch_data',
    python_callable=download_dataset,
    dag=dag,
)

extract_task = PythonOperator(
    task_id='unzip_data',
    python_callable=extract_dataset,
    dag=dag,
)

split_task = PythonOperator(
    task_id='prepare_splits',
    python_callable=split_data,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='push_to_minio',
    python_callable=upload_to_minio,
    dag=dag,
)

train_predict_task = PythonOperator(
    task_id='spark_train_predict',
    python_callable=train_and_predict,
    dag=dag,
)

download_task >> extract_task >> split_task >> upload_task >> train_predict_task
