# Databricks notebook source

# COMMAND ----------

from pyspark.sql import SparkSession

from airline_delay.schemas import ProjectConfig
from airline_delay.settings import PROJECT_CONFIG_LOCATION

project_config = ProjectConfig.from_yaml(config_path=PROJECT_CONFIG_LOCATION)
spark_session = SparkSession.builder.getOrCreate()

# COMMAND ----------

from airline_delay.data_processing.pipeline import DataPipeline

feature_pipeline = DataPipeline(config=project_config, spark=spark_session)

raw_data = feature_pipeline.extract_data()
processed_data = feature_pipeline.process_data(raw_data)
train_data, test_data = feature_pipeline.create_splits(processed_data)

feature_pipeline.save_data(train_data, target_table_name="train_data")
feature_pipeline.save_data(test_data, target_table_name="test_data")