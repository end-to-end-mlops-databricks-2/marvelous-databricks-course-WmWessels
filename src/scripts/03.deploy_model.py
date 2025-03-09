from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from airline_delay.schemas import ProjectConfig
from airline_delay.serving.model_serving import ModelServing
from airline_delay.settings import PROJECT_CONFIG_LOCATION

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

config = ProjectConfig.from_yaml(config_path=PROJECT_CONFIG_LOCATION)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name

model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.airline_delay_model_basic", endpoint_name="airline-delay-model-serving"
)

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

logger.info("Started deployment/update of the serving endpoint")