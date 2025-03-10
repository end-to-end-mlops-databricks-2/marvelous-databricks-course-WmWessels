import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from sklearn.model_selection import train_test_split

from airline_delay.schemas import ProjectConfig


class DataPipeline:
    def __init__(self, config: ProjectConfig, spark: SparkSession):
        self._config = config
        self._spark = spark

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._cast_type_to_categorical(df)
        df = self._cast_type_to_numerical(df)
        return df

    def create_splits(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2)
        return train_df, test_df

    def save_data(self, df: pd.DataFrame, target_table_name: str) -> pd.DataFrame:
        sdf = self._spark.createDataFrame(df)
        sdf = sdf.withColumn("update_timestamp_utc", sf.to_utc_timestamp(sf.current_timestamp(), "UTC"))
        sdf.write.mode("append").saveAsTable(
            f"{self._config.catalog_name}.{self._config.schema_name}.{target_table_name}"
        )

    def _cast_type_to_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        for numerical_column in self._config.numerical_features:
            df[numerical_column] = pd.to_numeric(df[numerical_column], errors="coerce").astype("float")
        return df

    def _cast_type_to_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        for categorical_column in self._config.categorical_features:
            df[categorical_column] = df[categorical_column].astype("category")
        return df


def create_synthetic_data(df: pd.DataFrame, num_rows=100):
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            synthetic_data[column] = np.random.randint(df[column].min(), df[column].max() + 1, num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    int_columns = {
        "_c0",
        "Delay",
    }
    for col in int_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.int32)

    return synthetic_data
