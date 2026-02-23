# SupplyGuard: AI-Powered Supply Chain Forecasting

## Overview
An end-to-end supply chain analytics and predictive risk modeling platform built on **Snowflake**, **XGBoost**, and **Streamlit**. Processes 358,000+ real-time logistical records to deliver proactive risk metrics and disruption forecasting with sub-second dashboard response times.

## Features

### Analytics Module
- **Unified Data Layer**: Merges raw API feeds into structured Snowflake datasets for systematic tracking and analysis of logistical records.
- **Dynamic Dashboards**: Streamlit-based risk visualization driving executive decision-making and operational planning, accelerating response by up to 48 hours.
- **Optimized SQL Views**: Complex multi-source data aggregation achieving sub-second query response for high-traffic dashboard components.

### Predictive Risk Modeling
- **XGBoost Disruption Forecasting**: Gradient-boosted decision trees forecasting supply chain disruptions across 358K+ records with optimized hyperparameters via randomized grid search.
- **Concurrent Data Ingestion**: Highly parallel Snowflake-to-training pipeline reducing model retraining time from 6 hours to 45 minutes.
- **High-Recall Detection**: Improved disruption prediction recall from 0.61 to 0.84 using advanced multi-variate time-series risk signals.

## Tech Stack
- **Data Warehouse**: Snowflake
- **ML Framework**: XGBoost, Scikit-Learn
- **Dashboard**: Streamlit
- **Language**: Python, SQL

## Project Structure
```
├── app.py                     # Streamlit dashboard application
├── src/
│   ├── data_pipeline.py       # Snowflake data ingestion & processing
│   └── forecasting_model.py   # XGBoost disruption prediction model
├── sql/
│   └── views.sql              # Optimized analytical SQL views
├── requirements.txt
└── README.md
```

## Getting Started
1. Configure Snowflake credentials in `.env`
2. Run `pip install -r requirements.txt`
3. Execute SQL views: `snowsql -f sql/views.sql`
4. Train model: `python src/forecasting_model.py`
5. Launch dashboard: `streamlit run app.py`
