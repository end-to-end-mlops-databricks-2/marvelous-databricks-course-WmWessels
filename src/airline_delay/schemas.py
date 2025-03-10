from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel


class Environment(Enum):
    LOCAL = "local"
    DATABRICKS = "databricks"


class ProjectConfig(BaseModel):
    numerical_features: list[str]
    categorical_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    experiment_name_basic: str | None = None
    experiment_name_fe: str | None = None
    env: Environment

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str
    job_run_id: str
