"""
SupplyGuard — Streamlit Risk Analytics Dashboard
Dynamic dashboards for proactive risk metrics, executive decision-making,
and operational planning acceleration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="SupplyGuard Analytics",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load logistical records from Snowflake (demo: synthetic data)."""
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2023-01-01", periods=n, freq="2h")
    df = pd.DataFrame({
        "shipment_id": [f"SH-{i:06d}" for i in range(n)],
        "ship_date": dates,
        "carrier": np.random.choice(["FedEx", "UPS", "DHL", "USPS"], n),
        "origin": np.random.choice(["US", "CN", "DE", "JP", "IN"], n),
        "destination": np.random.choice(["US", "UK", "DE", "FR", "CA"], n),
        "risk_score": np.random.beta(2, 5, n),
        "disrupted": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "weight_kg": np.random.uniform(0.5, 50, n),
        "transit_days": np.random.randint(1, 14, n),
    })
    return df


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.title("🛡️ SupplyGuard")
st.sidebar.markdown("---")

carriers = st.sidebar.multiselect("Carrier", df["carrier"].unique(), default=df["carrier"].unique())
origins = st.sidebar.multiselect("Origin", df["origin"].unique(), default=df["origin"].unique())
risk_threshold = st.sidebar.slider("Risk Score Threshold", 0.0, 1.0, 0.5)

filtered = df[
    (df["carrier"].isin(carriers))
    & (df["origin"].isin(origins))
]

# ---------------------------------------------------------------------------
# KPI header
# ---------------------------------------------------------------------------

st.title("Supply Chain Risk Analytics")
st.markdown("Real-time monitoring of 358,000+ logistical records")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Shipments", f"{len(filtered):,}")
col2.metric("Disruption Rate", f"{filtered['disrupted'].mean():.1%}")
col3.metric("Avg Risk Score", f"{filtered['risk_score'].mean():.3f}")
col4.metric("High-Risk Shipments", f"{(filtered['risk_score'] > risk_threshold).sum():,}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Disruption Rate by Carrier")
    carrier_stats = (
        filtered.groupby("carrier")["disrupted"]
        .mean()
        .reset_index()
        .rename(columns={"disrupted": "disruption_rate"})
    )
    fig = px.bar(
        carrier_stats, x="carrier", y="disruption_rate",
        color="disruption_rate", color_continuous_scale="Reds",
        labels={"disruption_rate": "Rate"},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Risk Score Distribution")
    fig2 = px.histogram(
        filtered, x="risk_score", nbins=50,
        color_discrete_sequence=["#636EFA"],
        labels={"risk_score": "Risk Score"},
    )
    fig2.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                   annotation_text="Threshold")
    st.plotly_chart(fig2, use_container_width=True)

# Daily disruption trend
st.subheader("Daily Disruption Trend")
daily = (
    filtered.set_index("ship_date")
    .resample("D")["disrupted"]
    .agg(["sum", "count"])
    .reset_index()
)
daily["rate"] = daily["sum"] / daily["count"]
fig3 = px.line(daily, x="ship_date", y="rate", labels={"rate": "Disruption Rate"})
fig3.update_traces(line_color="#EF553B")
st.plotly_chart(fig3, use_container_width=True)

# Risk heatmap by origin-destination
st.subheader("Risk Heatmap: Origin → Destination")
pivot = filtered.pivot_table(values="risk_score", index="origin", columns="destination", aggfunc="mean")
fig4 = px.imshow(pivot, color_continuous_scale="YlOrRd", labels={"color": "Avg Risk"})
st.plotly_chart(fig4, use_container_width=True)
