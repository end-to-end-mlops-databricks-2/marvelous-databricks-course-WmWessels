# Databricks notebook source
# !pip install /Volumes/mlops_dev/wichertw/packages/airline_delay-0.0.1-py3-none-any.whl

# COMMAND ----------
# %restart_python

# COMMAND ----------
import os
import time

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from airline_delay.schemas import ProjectConfig
from airline_delay.serving.model_serving import ModelServing
from airline_delay.settings import PROJECT_CONFIG_LOCATION

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path=PROJECT_CONFIG_LOCATION)
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.airline_delay_model_basic", endpoint_name="airline-delay-model-serving"
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = config.numerical_features + config.categorical_features

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_data").toPandas()

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------


def call_endpoint(self, record: list[dict]):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/airline-delay/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# "load test"

for i in range(len(dataframe_records)):
    call_endpoint(dataframe_records[i])
    time.sleep(0.2)
