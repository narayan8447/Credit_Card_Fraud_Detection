"""Streamlit app for credit card fraud detection."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import plotly.graph_objects as go
import streamlit as st

from utils import load_model_and_scaler, predict_transaction, prepare_features


APP_TITLE = "Credit Card Fraud Detector"
APP_SUBTITLE = "Hybrid Autoencoder + Random Forest model"


st.set_page_config(page_title=APP_TITLE, page_icon="💳", layout="centered")


@st.cache_resource(show_spinner="Loading trained model artifacts...")
def load_artifacts():
    """Cache ML artifacts so Streamlit does not reload them on every widget change."""
    return load_model_and_scaler()


def render_header() -> None:
    st.markdown(
        f"""
        <div style="text-align:center; padding-bottom: 0.75rem;">
            <h1 style="margin-bottom:0.25rem;">💳 {APP_TITLE}</h1>
            <p style="color:#6b7280; font-size:1.05rem;">{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_gauge(fraud_percent: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fraud_percent,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Fraud Probability", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#dc2626"},
                "steps": [
                    {"range": [0, 30], "color": "#bbf7d0"},
                    {"range": [30, 60], "color": "#fde68a"},
                    {"range": [60, 100], "color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.75,
                    "value": fraud_percent,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=24, r=24, t=48, b=12))
    return fig


def render_result(label: str, probability: float) -> None:
    fraud_percent = probability * 100
    st.subheader("Prediction Result")
    st.plotly_chart(build_gauge(fraud_percent), use_container_width=True)

    if label == "Fraudulent":
        st.error(f"Fraudulent transaction detected. Probability: {fraud_percent:.2f}%")
    else:
        st.success(f"Transaction looks normal. Probability: {fraud_percent:.2f}%")

    if probability < 0.30:
        st.markdown("### Low Risk")
    elif probability < 0.60:
        st.markdown("### Medium Risk")
    else:
        st.markdown("### High Risk")


def main() -> None:
    render_header()

    try:
        model, scaler, encoder = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Transaction Inputs")

    time_seconds = st.slider("Time elapsed in seconds", 0.0, 200000.0, 50000.0, step=100.0)
    amount = st.slider("Transaction amount", 0.0, 5000.0, 200.0, step=10.0)

    col1, col2 = st.columns(2)
    with col1:
        is_online = int(st.toggle("Online transaction"))
        attempts = int(st.toggle("Multiple attempts"))

    with col2:
        location_changed = int(st.toggle("Location changed"))
        card_type = int(st.toggle("Credit card", value=True))

    if st.button("Predict Fraud", use_container_width=True, type="primary"):
        base, context = prepare_features(
            time_seconds,
            amount,
            is_online,
            location_changed,
            attempts,
            card_type,
        )
        result = predict_transaction(model, scaler, encoder, base, context)
        render_result(result["label"], result["probability"])

        st.subheader("Transaction Summary")
        st.dataframe(
            {
                "Feature": [
                    "Time",
                    "Amount",
                    "Online",
                    "Location Changed",
                    "Multiple Attempts",
                    "Card Type",
                ],
                "Value": [
                    f"{time_seconds:,.0f} seconds",
                    f"${amount:,.2f}",
                    "Yes" if is_online else "No",
                    "Yes" if location_changed else "No",
                    "Yes" if attempts else "No",
                    "Credit" if card_type else "Debit",
                ],
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Adjust the transaction inputs, then click Predict Fraud.")


if __name__ == "__main__":
    main()
