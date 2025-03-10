# COMMAND ----------

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

config = ProjectConfig.from_yaml(config_path=PROJECT_CONFIG_LOCATION)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": "sha123", "branch": "feature-branch", "job_run_id": "123"}
tags = Tags(**tags_dict)

# Initialize model
basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

basic_model.load_data()
basic_model.prepare_features()
basic_model.train()
basic_model.log_model()
basic_model.register_model()
