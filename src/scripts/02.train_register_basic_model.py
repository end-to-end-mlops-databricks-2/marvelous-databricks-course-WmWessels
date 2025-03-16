import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from airline_delay.modeling.basic_model import BasicModel
from airline_delay.schemas import ProjectConfig, Tags
from airline_delay.settings import PROJECT_CONFIG_LOCATION

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()

config = ProjectConfig.from_yaml(config_path=PROJECT_CONFIG_LOCATION)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

basic_model.load_data()
basic_model.prepare_features()
basic_model.train()
basic_model.log_model()
model_improved = basic_model.model_improved()

if model_improved:
    # Register the model
    latest_version = basic_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
