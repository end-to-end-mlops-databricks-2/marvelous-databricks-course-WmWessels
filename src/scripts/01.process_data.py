from pyspark.sql import SparkSession

from airline_delay.schemas import ProjectConfig
from airline_delay.settings import PROJECT_CONFIG_LOCATION

import yaml
from loguru import logger

from airline_delay.data_processing.pipeline import DataPipeline, create_synthetic_data


project_config = ProjectConfig.from_yaml(config_path=PROJECT_CONFIG_LOCATION)
spark_session = SparkSession.builder.getOrCreate()

logger.info("Configuration loaded:")
logger.info(yaml.dump(project_config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(f"/Volumes/{project_config.catalog_name}/{project_config.schema_name}/data/airlines.csv", header=True, inferSchema=True).toPandas()

feature_pipeline = DataPipeline(config=project_config, spark=spark_session)
synthetic_df = create_synthetic_data(df)

processed_data = feature_pipeline.process_data(synthetic_df)
train_data, test_data = feature_pipeline.create_splits(processed_data)

feature_pipeline.save_data(train_data, target_table_name="train_data")
feature_pipeline.save_data(test_data, target_table_name="test_data")