
import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = 'plotly_dark'

RISK_COLORS = {
    "HIGH RISK": "#F43F5E",
    "MEDIUM RISK": "#F59E0B",
    "LOW RISK": "#10B981"
}
import shap
import networkx as nx

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🛡️",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: #0B1120;
    color: #E5E7EB;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}
[data-testid="stHeader"] {
    background: rgba(11,17,32,0.95);
}
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700;
}
p, label, span {
    color: #CBD5E1;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243A, #111827);
    border: 1px solid #263B58;
    padding: 20px;
    border-radius: 14px;
}
[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #263449;
    border-radius: 12px;
}
div[data-testid="stPlotlyChart"] {
    background: #111827;
    border: 1px solid #263449;
    border-radius: 14px;
    padding: 10px;
}
button[kind="secondary"] {
    border-radius: 8px;
}
hr {
    border-color: #263449;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="
    background: linear-gradient(110deg, #111827, #142B46);
    border: 1px solid #2563EB;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
">
    <div style="color:#38BDF8; font-size:13px;
                letter-spacing:3px; font-weight:700;">
        FINANCIAL INTELLIGENCE PLATFORM
    </div>
    <div style="font-size:34px; font-weight:800;
                color:#F8FAFC; margin-top:8px;">
        🛡️ FRAUDLENS AI
    </div>
    <div style="color:#94A3B8; font-size:15px; margin-top:5px;">
        Intelligent Fraud Detection • Explainable AI • Account Investigation
    </div>
</div>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# --------------------------------------------------
# LOAD PROJECT ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        f"{ARTIFACT_DIR}/fraudlens_transaction_model.pkl"
    )

    dashboard = pd.read_parquet(
        f"{ARTIFACT_DIR}/fraudlens_dashboard_data.parquet"
    )

    features = pd.read_parquet(
        f"{ARTIFACT_DIR}/fraudlens_test_features.parquet"
    )

    with open(
        f"{ARTIFACT_DIR}/model_metadata.json"
    ) as file:
        metadata = json.load(file)

    return model, dashboard, features, metadata


try:
    model, dashboard, features, metadata = load_artifacts()
except Exception as error:
    st.error(f"Could not load project artifacts: {error}")
    st.stop()

# Align data by transaction index
dashboard = dashboard.loc[features.index]

# Ensure the dashboard never displays ground-truth labels
dashboard = dashboard.drop(
    columns=["isFraud", "isFlaggedFraud"],
    errors="ignore"
)

# --------------------------------------------------
# RISK CLASSIFICATION
# --------------------------------------------------

dashboard["risk_category"] = np.select(
    [
        dashboard["risk_score"] >= 0.80,
        dashboard["risk_score"] >= 0.40
    ],
    [
        "HIGH RISK",
        "MEDIUM RISK"
    ],
    default="LOW RISK"
)

dashboard["risk_percentage"] = (
    dashboard["risk_score"] * 100
).round(2)

# --------------------------------------------------
# SHAP EXPLAINER
# --------------------------------------------------

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


explainer = load_explainer(model)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("""
<div style="padding:18px 4px 24px 4px;">
    <div style="font-family:Orbitron,sans-serif;
                font-size:22px;font-weight:800;
                color:#48E8FF;letter-spacing:2px;">
        ◈ FRAUDLENS
    </div>
    <div style="font-size:10px;letter-spacing:3px;
                color:#7189A5;margin-top:5px;">
        INTELLIGENCE CORE // V1.0
    </div>
    <div style="margin-top:22px;padding:10px;
                border:1px solid #164E63;
                border-radius:8px;background:#0C1D2B;">
        <span style="color:#34D399;">●</span>
        <span style="color:#A5F3FC;font-size:11px;
                     letter-spacing:1px;">
            SYSTEM OPERATIONAL
        </span>
    </div>
</div>
<div style="font-size:10px;letter-spacing:2px;
            color:#64748B;margin-bottom:12px;">
    CONTROL CENTER / MODULES
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "SELECT MODULE",
    [
        "Executive Overview",
        "Transaction Explorer",
        "AI Explainability",
        "Account Investigation"
    ],
    format_func=lambda x: {
        "Executive Overview": "◈  Command Overview",
        "Transaction Explorer": "⌕  Transaction Scanner",
        "AI Explainability": "✧  Neural Explainability",
        "Account Investigation": "◎  Network Intelligence"
    }[x]
)

st.sidebar.markdown("""
<div style="position:relative;margin-top:35px;
            padding-top:15px;border-top:1px solid #24364D;
            color:#64748B;font-size:10px;letter-spacing:1px;">
    FRAUD DETECTION ENGINE<br>
    PAYSim // SYNTHETIC DATASET<br><br>
    BUILD 01.0.0 | XGBOOST + SHAP
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# EXECUTIVE OVERVIEW
# --------------------------------------------------

if page == "Executive Overview":

    st.header("Executive Overview")

    total = len(dashboard)
    high = int((dashboard["risk_category"] == "HIGH RISK").sum())
    medium = int((dashboard["risk_category"] == "MEDIUM RISK").sum())
    low = int((dashboard["risk_category"] == "LOW RISK").sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions Analysed", f"{total:,}")
    c2.metric("High Risk", f"{high:,}")
    c3.metric("Medium Risk", f"{medium:,}")
    c4.metric("Low Risk", f"{low:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        risk_counts = pd.DataFrame({
            "Risk Category": ["HIGH RISK", "MEDIUM RISK", "LOW RISK"],
            "Transactions": [high, medium, low]
        })

        fig = px.pie(
            risk_counts,
            names="Risk Category",
            values="Transactions",
            hole=0.55,
            title="Transaction Risk Distribution"
        )

        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#E2E8F0",
            title_font_size=17,
            margin=dict(l=20, r=20, t=55, b=25)
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        type_summary = (
            dashboard.groupby("type")
            .size()
            .reset_index(name="Transactions")
        )

        fig = px.bar(
            type_summary,
            x="type",
            y="Transactions",
            title="Transactions by Type"
        )

        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#E2E8F0",
            title_font_size=17,
            margin=dict(l=20, r=20, t=55, b=25)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-Risk Transactions")

    top = dashboard.nlargest(10, "risk_score")

    st.dataframe(
        top[
            [
                "type",
                "amount",
                "risk_percentage",
                "risk_category"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# TRANSACTION EXPLORER
# --------------------------------------------------

elif page == "Transaction Explorer":

    st.header("Transaction Explorer")

    risk_filter = st.multiselect(
        "Filter by risk category",
        ["HIGH RISK", "MEDIUM RISK", "LOW RISK"],
        default=["HIGH RISK", "MEDIUM RISK", "LOW RISK"]
    )

    type_filter = st.multiselect(
        "Filter by transaction type",
        sorted(dashboard["type"].dropna().unique()),
        default=sorted(dashboard["type"].dropna().unique())
    )

    minimum_amount = float(dashboard["amount"].min())
    maximum_amount = float(dashboard["amount"].max())

    amount_range = st.slider(
        "Transaction amount range",
        min_value=minimum_amount,
        max_value=maximum_amount,
        value=(minimum_amount, maximum_amount)
    )

    filtered = dashboard[
        dashboard["risk_category"].isin(risk_filter)
        & dashboard["type"].isin(type_filter)
        & dashboard["amount"].between(*amount_range)
    ]

    st.metric("Matching Transactions", f"{len(filtered):,}")

    st.dataframe(
        filtered[
            [
                "nameOrig",
                "nameDest",
                "type",
                "amount",
                "risk_percentage",
                "risk_category"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# AI EXPLAINABILITY
# --------------------------------------------------

elif page == "AI Explainability":

    st.header("AI Explainability")
    st.write("Understand the features influencing a transaction's model risk score.")

    transaction_id = st.number_input(
        "Enter transaction index",
        min_value=int(dashboard.index.min()),
        max_value=int(dashboard.index.max()),
        value=int(dashboard.nlargest(1, "risk_score").index[0]),
        step=1
    )

    if transaction_id in features.index:

        row = features.loc[[transaction_id]]

        probability = float(
            model.predict_proba(row)[0, 1]
        )

        shap_values = explainer.shap_values(row)

        category = (
            "HIGH RISK" if probability >= 0.80
            else "MEDIUM RISK" if probability >= 0.40
            else "LOW RISK"
        )

        c1, c2 = st.columns(2)

        c1.metric("Predicted Fraud Risk", f"{probability:.2%}")
        c2.metric("Risk Category", category)

        explanation = pd.DataFrame({
            "Feature": row.columns,
            "Value": row.iloc[0].values,
            "SHAP Contribution": shap_values[0]
        })

        explanation["Direction"] = np.where(
            explanation["SHAP Contribution"] > 0,
            "Increases Risk",
            "Decreases Risk"
        )

        explanation["Absolute Impact"] = (
            explanation["SHAP Contribution"].abs()
        )

        explanation = explanation.sort_values(
            "Absolute Impact",
            ascending=True
        )

        fig = px.bar(
            explanation,
            x="SHAP Contribution",
            y="Feature",
            color="Direction",
            orientation="h",
            title="Individual Transaction Explanation"
        )

        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#E2E8F0",
            title_font_size=17,
            margin=dict(l=20, r=20, t=55, b=25)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            explanation.sort_values(
                "Absolute Impact",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "SHAP values explain model behaviour. "
            "They do not establish that a transaction is actually fraudulent."
        )

    else:
        st.warning("Transaction index not found.")

# --------------------------------------------------
# ACCOUNT INVESTIGATION
# --------------------------------------------------

elif page == "Account Investigation":

    st.header("Account Relationship Investigation")

    account_type = st.radio(
        "Account role",
        ["Sender", "Receiver"],
        horizontal=True
    )

    account_column = (
        "nameOrig" if account_type == "Sender"
        else "nameDest"
    )

    account_options = sorted(
        dashboard[account_column].dropna().unique().tolist()
    )

    selected_account = st.selectbox(
        "Select account",
        account_options
    )

    account_transactions = dashboard[
        (dashboard["nameOrig"] == selected_account)
        | (dashboard["nameDest"] == selected_account)
    ]

    st.metric(
        "Related Transactions",
        f"{len(account_transactions):,}"
    )

    st.dataframe(
        account_transactions.sort_values(
            "risk_score",
            ascending=False
        )[
            [
                "nameOrig",
                "nameDest",
                "type",
                "amount",
                "risk_percentage",
                "risk_category"
            ]
        ].head(100),
        use_container_width=True,
        hide_index=True
    )

    # Construct a local account relationship graph
    G = nx.Graph()

    for _, transaction in account_transactions.head(200).iterrows():

        sender = transaction["nameOrig"]
        receiver = transaction["nameDest"]

        G.add_edge(
            sender,
            receiver,
            amount=float(transaction["amount"]),
            risk=float(transaction["risk_score"])
        )

    if G.number_of_edges() > 0:

        positions = nx.spring_layout(
            G,
            seed=42,
            k=0.6
        )

        edge_x = []
        edge_y = []

        for source, target in G.edges():
            x0, y0 = positions[source]
            x1, y1 = positions[target]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1),
            hoverinfo="none"
        )

        node_x = []
        node_y = []
        node_text = []

        for node in G.nodes():
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=node_text,
            hoverinfo="text",
            marker=dict(size=10)
        )

        fig = go.Figure(
            data=[edge_trace, node_trace]
        )

        fig.update_layout(
            title="Account Transaction Network",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#E2E8F0",
            title_font_size=17,
            margin=dict(l=20, r=20, t=55, b=25)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No account relationships available.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FraudLens AI | Research prototype using synthetic PaySim data. "
    "Predictions are risk indicators, not proof of fraud."
)



# FRAUDLENS_CYBER_THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
      radial-gradient(ellipse at 10% 0%, rgba(0,220,255,.10), transparent 35%),
      radial-gradient(ellipse at 95% 10%, rgba(139,92,246,.12), transparent 35%),
      #070B14;
    color: #E5F3FF;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#090F1D,#101329 65%,#0B1020);
    border-right: 1px solid rgba(0,229,255,.25);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    font-family: 'Orbitron', sans-serif;
    color: #54E8FF !important;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: .5px;
    color: #EAF8FF !important;
}

h1 {
    text-shadow: 0 0 18px rgba(0,220,255,.35);
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg,rgba(17,31,53,.96),rgba(12,17,32,.96));
    border: 1px solid rgba(0,220,255,.28);
    border-radius: 13px;
    padding: 19px;
    box-shadow: 0 0 18px rgba(0,180,255,.06);
}

[data-testid="stMetricValue"] {
    color: #53E8FF !important;
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0 0 12px rgba(0,220,255,.25);
}

[data-testid="stMetricLabel"] {
    color: #8FAAC4 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(13,22,39,.86);
    border: 1px solid rgba(74,163,255,.22);
    border-radius: 15px;
    padding: 12px;
    box-shadow: 0 0 20px rgba(0,150,255,.05);
}

.stButton > button {
    background: linear-gradient(100deg,#087EAA,#5141A8);
    color: white !important;
    border: 1px solid #42DFFF;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button:hover {
    box-shadow: 0 0 18px rgba(0,220,255,.35);
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,220,255,.20);
    border-radius: 12px;
}

hr {
    border-color: rgba(0,220,255,.22);
}
</style>
""", unsafe_allow_html=True)
