FROM apache/airflow:2.10.3

RUN pip install pyspark==3.5.0
RUN pip install pandas
RUN pip install minio
RUN pip install pyarrow

USER root
RUN apt-get update && apt-get install -y default-jdk && apt-get autoremove -yqq --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
