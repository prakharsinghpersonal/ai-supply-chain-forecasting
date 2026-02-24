"""
SupplyGuard Configuration
=========================
Configuration module for the supply chain analytics and risk modeling pipeline.
"""
import os
from dataclasses import dataclass


@dataclass
class SnowflakeConfig:
    """Snowflake connection configuration."""
    account: str = os.getenv("SNOWFLAKE_ACCOUNT", "")
    user: str = os.getenv("SNOWFLAKE_USER", "")
    password: str = os.getenv("SNOWFLAKE_PASSWORD", "")
    warehouse: str = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    database: str = os.getenv("SNOWFLAKE_DATABASE", "SUPPLYGUARD")
    schema: str = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
    role: str = os.getenv("SNOWFLAKE_ROLE", "SYSADMIN")


@dataclass
class ModelConfig:
    """ML model configuration for disruption prediction."""
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 50
    random_state: int = 42


@dataclass
class AppConfig:
    """Streamlit/Power BI dashboard configuration."""
    page_title: str = "SupplyGuard Analytics"
    refresh_interval: int = 300  # seconds
    max_records_display: int = 10000
    cache_ttl: int = 600
    snowflake: SnowflakeConfig = None
    model: ModelConfig = None

    def __post_init__(self):
        if self.snowflake is None:
            self.snowflake = SnowflakeConfig()
        if self.model is None:
            self.model = ModelConfig()
