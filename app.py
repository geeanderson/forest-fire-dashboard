import os
import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

AWS_REGION     = _secret("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = _secret("DYNAMODB_TABLE", "forest-fire-readings")
NODES         = ["node-01", "node-02", "node-03"]
RISK_LABELS   = {0: "Nenhum", 1: "Baixo", 2: "Moderado", 3: "Alto", 4: "Muito Alto", 5: "Crítico"}
RISK_COLORS   = {0: "green", 1: "lightgreen", 2: "yellow", 3: "orange", 4: "red", 5: "darkred"}

st.set_page_config(
    page_title="Forest Fire IoT Monitor",
    page_icon="🔥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def get_table():
    kwargs = {"region_name": AWS_REGION}
    key_id = _secret("AWS_ACCESS_KEY_ID")
    secret_key = _secret("AWS_SECRET_ACCESS_KEY")
    if key_id and secret_key:
        kwargs["aws_access_key_id"] = key_id
        kwargs["aws_secret_access_key"] = secret_key
    dynamodb = boto3.resource("dynamodb", **kwargs)
    return dynamodb.Table(DYNAMODB_TABLE)


def load_readings(hours: int = 24) -> pd.DataFrame:
    table = get_table()
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    frames = []
    for node in NODES:
        resp = table.query(
            KeyConditionExpression=Key("device_id").eq(node) & Key("timestamp").gte(since),
            ScanIndexForward=True,
        )
        if resp["Items"]:
            frames.append(pd.DataFrame(resp["Items"]))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    numeric_cols = ["temperature", "humidity", "air_quality", "fire_risk",
                    "fire_probability", "days_without_rain", "battery_voltage"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🔥 Forest Fire IoT Monitor")
st.caption(f"Tabela DynamoDB: `{DYNAMODB_TABLE}` · Região: `{AWS_REGION}`")

col_refresh, col_window = st.columns([1, 3])
with col_refresh:
    if st.button("↻ Atualizar"):
        st.cache_data.clear()
with col_window:
    hours = st.slider("Janela de tempo (horas)", 1, 168, 24)

df = load_readings(hours)

if df.empty:
    st.warning("Nenhuma leitura encontrada no período selecionado.")
    st.stop()

# ---------------------------------------------------------------------------
# Cards — última leitura por nó
# ---------------------------------------------------------------------------

st.subheader("Última leitura por nó")
cols = st.columns(len(NODES))

for i, node in enumerate(NODES):
    node_df = df[df["device_id"] == node]
    if node_df.empty:
        cols[i].info(f"{node}\nSem dados")
        continue

    last = node_df.iloc[-1]
    risk = int(last.get("fire_risk", -1)) if pd.notna(last.get("fire_risk")) else -1
    risk_label = RISK_LABELS.get(risk, "N/A")
    risk_color = RISK_COLORS.get(risk, "gray")

    with cols[i]:
        st.markdown(f"**{node}**")
        st.metric("Temperatura (°C)", f"{last.get('temperature', 0):.1f}")
        st.metric("Umidade (%)", f"{last.get('humidity', 0):.1f}")
        st.metric("Qualidade do Ar", f"{last.get('air_quality', 0):.0f}")
        days = last.get("days_without_rain")
        if pd.notna(days):
            st.metric("Dias sem chuva", f"{int(days)}")
        prob = last.get("fire_probability")
        prob_str = f"{float(prob)*100:.1f}%" if pd.notna(prob) else "N/A"
        st.markdown(
            f"<div style='background:{risk_color};padding:8px;border-radius:4px;"
            f"text-align:center;color:white;font-weight:bold;margin-top:4px'>"
            f"Risco: {risk_label} ({risk}/5)<br>"
            f"<small>Probabilidade: {prob_str}</small></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Gráficos de séries temporais
# ---------------------------------------------------------------------------

st.subheader("Séries temporais")

tab_temp, tab_hum, tab_aq, tab_risk, tab_prob = st.tabs(
    ["🌡️ Temperatura", "💧 Umidade", "💨 Qualidade do Ar", "🔥 Risco (0–5)", "📊 Probabilidade ML"]
)

with tab_temp:
    fig = px.line(df, x="timestamp", y="temperature", color="device_id",
                  labels={"temperature": "°C", "timestamp": "", "device_id": "Nó"})
    st.plotly_chart(fig, use_container_width=True)

with tab_hum:
    fig = px.line(df, x="timestamp", y="humidity", color="device_id",
                  labels={"humidity": "%", "timestamp": "", "device_id": "Nó"})
    st.plotly_chart(fig, use_container_width=True)

with tab_aq:
    fig = px.line(df, x="timestamp", y="air_quality", color="device_id",
                  labels={"air_quality": "ADC raw", "timestamp": "", "device_id": "Nó"})
    st.plotly_chart(fig, use_container_width=True)

with tab_risk:
    if "fire_risk" in df.columns:
        fig = px.line(df, x="timestamp", y="fire_risk", color="device_id",
                      labels={"fire_risk": "Nível (0–5)", "timestamp": "", "device_id": "Nó"},
                      range_y=[0, 5])
        fig.add_hrect(y0=3, y1=5, fillcolor="red", opacity=0.1, line_width=0,
                      annotation_text="Zona de risco")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Escala: 0=Nenhum · 1=Baixo · 2=Moderado · 3=Alto · 4=Muito Alto · 5=Crítico")
    else:
        st.info("Coluna fire_risk não encontrada nos dados.")

with tab_prob:
    if "fire_probability" in df.columns:
        fig = px.line(df, x="timestamp", y="fire_probability", color="device_id",
                      labels={"fire_probability": "Probabilidade", "timestamp": "", "device_id": "Nó"},
                      range_y=[0, 1])
        fig.add_hrect(y0=0.5, y1=1, fillcolor="orange", opacity=0.1, line_width=0,
                      annotation_text="P > 50%")
        fig.add_hrect(y0=0.7, y1=1, fillcolor="red", opacity=0.1, line_width=0,
                      annotation_text="P > 70%")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Probabilidade de incêndio prevista pelo Random Forest (F1=0.748 / AUC=0.801)")
    else:
        st.info("Coluna fire_probability não encontrada nos dados.")

st.divider()

# ---------------------------------------------------------------------------
# Tabela de leituras recentes
# ---------------------------------------------------------------------------

st.subheader("Leituras recentes")
show_cols = [c for c in [
    "device_id", "timestamp", "temperature", "humidity", "air_quality",
    "days_without_rain", "fire_risk", "fire_probability", "battery_voltage",
] if c in df.columns]
recent_df = df[show_cols].tail(50).sort_values("timestamp", ascending=False)
st.dataframe(recent_df, use_container_width=True)

csv_data = df[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Exportar CSV",
    data=csv_data,
    file_name=f"forest_fire_readings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)
