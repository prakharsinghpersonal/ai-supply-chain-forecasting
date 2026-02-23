"""
SupplyGuard — XGBoost Disruption Forecasting Model
Gradient-boosted decision trees to predict supply chain disruptions,
with randomized hyperparameter search and multi-variate time-series features.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer multi-variate time-series risk features from raw shipment and
    supplier data for disruption prediction.
    """
    df = df.copy()

    # Temporal features
    df["ship_dayofweek"] = pd.to_datetime(df["ship_date"]).dt.dayofweek
    df["ship_month"] = pd.to_datetime(df["ship_date"]).dt.month
    df["transit_days"] = (
        pd.to_datetime(df["expected_delivery"]) - pd.to_datetime(df["ship_date"])
    ).dt.days

    # Rolling risk signals (7-day & 30-day windows)
    for window in [7, 30]:
        df[f"rolling_{window}d_risk_mean"] = (
            df.sort_values("ship_date")
            .groupby("carrier")["risk_score"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"rolling_{window}d_risk_std"] = (
            df.sort_values("ship_date")
            .groupby("carrier")["risk_score"]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # Supplier reliability lag features
    if "on_time_rate" in df.columns:
        df["supplier_reliability_lag1"] = df.groupby("supplier_id")[
            "on_time_rate"
        ].shift(1)
        df["supplier_defect_trend"] = df.groupby("supplier_id")[
            "defect_rate"
        ].diff()

    df.fillna(0, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
    "scale_pos_weight": [1, 3, 5, 10],
}


def train_model(
    df: pd.DataFrame,
    target_col: str = "disrupted",
    n_iter: int = 50,
    test_size: float = 0.2,
) -> dict:
    """
    Train an XGBoost classifier with randomized hyperparameter search.

    Returns
    -------
    dict – metrics and path to persisted model artefact.
    """
    feature_cols = [
        c for c in df.columns
        if c not in [target_col, "shipment_id", "ship_date", "expected_delivery"]
    ]

    # Encode categorical columns
    encoders = {}
    for col in df[feature_cols].select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info("Training set: %d | Test set: %d", len(X_train), len(X_test))

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
    )

    search = RandomizedSearchCV(
        base_model,
        PARAM_GRID,
        n_iter=n_iter,
        scoring="recall",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "best_params": search.best_params_,
    }

    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    logger.info("Metrics: %s", metrics)

    # Persist model
    model_path = MODEL_DIR / "xgb_disruption_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "encoders": encoders, "features": feature_cols}, f)
    logger.info("Model saved to %s", model_path)

    metrics["model_path"] = str(model_path)
    return metrics


if __name__ == "__main__":
    # Demo: generate synthetic data for local testing
    np.random.seed(42)
    n = 10000
    demo_df = pd.DataFrame({
        "shipment_id": [f"SH-{i:06d}" for i in range(n)],
        "carrier": np.random.choice(["FedEx", "UPS", "DHL", "USPS"], n),
        "origin": np.random.choice(["US", "CN", "DE", "JP", "IN"], n),
        "destination": np.random.choice(["US", "UK", "DE", "FR", "CA"], n),
        "ship_date": pd.date_range("2022-01-01", periods=n, freq="h"),
        "expected_delivery": pd.date_range("2022-01-04", periods=n, freq="h"),
        "weight_kg": np.random.uniform(0.5, 50, n),
        "risk_score": np.random.uniform(0, 1, n),
        "disrupted": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })
    demo_df = build_features(demo_df)
    train_model(demo_df)
