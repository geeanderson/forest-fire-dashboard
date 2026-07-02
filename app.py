import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
import plotly.express as px
import streamlit as st
from boto3.dynamodb.conditions import Key

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
NODES          = ["node-01", "node-02", "node-03"]
RISK_LABELS    = {0: "Nenhum", 1: "Baixo", 2: "Moderado", 3: "Alto", 4: "Muito Alto", 5: "Crítico"}
RISK_COLORS    = {0: "#2e7d32", 1: "#66bb6a", 2: "#f9a825", 3: "#ef6c00", 4: "#c62828", 5: "#4a148c"}

PRESETS = {
    "Últimos 15min":  timedelta(minutes=15),
    "Última 1h":      timedelta(hours=1),
    "Últimas 6h":     timedelta(hours=6),
    "Últimas 24h":    timedelta(hours=24),
    "Últimos 7d":     timedelta(days=7),
    "Últimos 30d":    timedelta(days=30),
    "Últimos 365d":   timedelta(days=365),
    "Personalizado":  None,
}

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
    return boto3.resource("dynamodb", **kwargs).Table(DYNAMODB_TABLE)


def load_readings(start: datetime, end: datetime) -> pd.DataFrame:
    table = get_table()
    since = start.isoformat()
    until = end.isoformat()

    frames = []
    for node in NODES:
        resp = table.query(
            KeyConditionExpression=(
                Key("device_id").eq(node)
                & Key("timestamp").between(since, until)
            ),
            ScanIndexForward=True,
        )
        items = resp["Items"]
        while "LastEvaluatedKey" in resp:
            resp = table.query(
                KeyConditionExpression=(
                    Key("device_id").eq(node)
                    & Key("timestamp").between(since, until)
                ),
                ExclusiveStartKey=resp["LastEvaluatedKey"],
                ScanIndexForward=True,
            )
            items.extend(resp["Items"])
        if items:
            frames.append(pd.DataFrame(items))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in ["temperature", "humidity", "air_quality", "fire_risk",
                "fire_probability", "days_without_rain", "battery_voltage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔥 Forest Fire IoT Monitor")
st.caption(f"DynamoDB: `{DYNAMODB_TABLE}` · Região: `{AWS_REGION}`")

# ---------------------------------------------------------------------------
# Time range picker (estilo Grafana)
# ---------------------------------------------------------------------------

with st.container():
    col_preset, col_refresh = st.columns([5, 1])

    with col_preset:
        preset = st.radio(
            "Intervalo de tempo",
            options=list(PRESETS.keys()),
            index=3,          # Últimas 24h como padrão
            horizontal=True,
            label_visibility="collapsed",
        )

    with col_refresh:
        refresh = st.button("↻ Atualizar", use_container_width=True)
        if refresh:
            st.cache_data.clear()

now_utc = datetime.now(timezone.utc)

if PRESETS[preset] is not None:
    start_dt = now_utc - PRESETS[preset]
    end_dt   = now_utc
    st.caption(f"📅 {start_dt.strftime('%Y-%m-%d %H:%M UTC')}  →  {end_dt.strftime('%Y-%m-%d %H:%M UTC')}")
else:
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Data inicial", value=(now_utc - timedelta(days=1)).date())
        start_time = st.time_input("Hora inicial", value=datetime.min.time())
    with col_e:
        end_date = st.date_input("Data final", value=now_utc.date())
        end_time = st.time_input("Hora final", value=datetime.max.time().replace(second=0, microsecond=0))
    start_dt = datetime.combine(start_date, start_time, tzinfo=timezone.utc)
    end_dt   = datetime.combine(end_date, end_time, tzinfo=timezone.utc)
    st.caption(f"📅 {start_dt.strftime('%Y-%m-%d %H:%M UTC')}  →  {end_dt.strftime('%Y-%m-%d %H:%M UTC')}")

df = load_readings(start_dt, end_dt)

if df.empty:
    st.warning(
        f"Nenhuma leitura encontrada entre "
        f"**{start_dt.strftime('%Y-%m-%d %H:%M UTC')}** e "
        f"**{end_dt.strftime('%Y-%m-%d %H:%M UTC')}**."
    )
    st.info("Dica: use 'Personalizado' para selecionar intervalos anteriores, como 2026-05-31.")
    st.stop()

st.caption(f"**{len(df)} leituras** de {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} a {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')} UTC")

st.divider()

# ---------------------------------------------------------------------------
# Cards — última leitura por nó
# ---------------------------------------------------------------------------

st.subheader("Última leitura por nó")
cols = st.columns(len(NODES))

for i, node in enumerate(NODES):
    node_df = df[df["device_id"] == node]
    if node_df.empty:
        with cols[i]:
            st.info(f"**{node}**\nSem dados no período")
        continue

    last = node_df.iloc[-1]
    risk = int(last.get("fire_risk", -1)) if pd.notna(last.get("fire_risk")) else -1
    risk_label = RISK_LABELS.get(risk, "N/A")
    risk_color = RISK_COLORS.get(risk, "#555")
    source = str(last.get("source", ""))
    source_tag = f" `{source}`" if source else ""

    with cols[i]:
        st.markdown(f"**{node}**{source_tag}  \n`{last['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}`")
        st.metric("Temperatura (°C)", f"{last.get('temperature', 0):.1f}")
        st.metric("Umidade (%)", f"{last.get('humidity', 0):.1f}")
        st.metric("Qualidade do Ar", f"{last.get('air_quality', 0):.0f}")
        days = last.get("days_without_rain")
        if pd.notna(days):
            st.metric("Dias sem chuva", int(days))
        prob = last.get("fire_probability")
        prob_str = f"{float(prob)*100:.1f}%" if pd.notna(prob) else "N/A"
        st.markdown(
            f"<div style='background:{risk_color};padding:8px;border-radius:6px;"
            f"text-align:center;color:white;font-weight:bold;margin-top:6px'>"
            f"Risco: {risk_label} ({risk}/5)<br>"
            f"<small>Probabilidade: {prob_str}</small></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Séries temporais
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
        st.caption("0=Nenhum · 1=Baixo · 2=Moderado · 3=Alto · 4=Muito Alto · 5=Crítico")
    else:
        st.info("Coluna fire_risk não encontrada.")

with tab_prob:
    if "fire_probability" in df.columns:
        fig = px.line(df, x="timestamp", y="fire_probability", color="device_id",
                      labels={"fire_probability": "Probabilidade", "timestamp": "", "device_id": "Nó"},
                      range_y=[0, 1])
        fig.add_hrect(y0=0.5, y1=1.0, fillcolor="orange", opacity=0.08, line_width=0,
                      annotation_text="P > 50%")
        fig.add_hrect(y0=0.7, y1=1.0, fillcolor="red", opacity=0.08, line_width=0,
                      annotation_text="P > 70%")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Random Forest · F1=0.748 · AUC=0.801 · Features: temperatura, umidade, dias sem chuva")
    else:
        st.info("Coluna fire_probability não encontrada.")

st.divider()

# ---------------------------------------------------------------------------
# Tabela de leituras
# ---------------------------------------------------------------------------

st.subheader("Leituras")

show_cols = [c for c in [
    "device_id", "timestamp", "temperature", "humidity", "air_quality",
    "days_without_rain", "fire_risk", "fire_probability", "source", "battery_voltage",
] if c in df.columns]

st.dataframe(
    df[show_cols].sort_values("timestamp", ascending=False),
    use_container_width=True,
    height=300,
)

csv_data = df[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Exportar CSV",
    data=csv_data,
    file_name=f"forest_fire_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv",
    mime="text/csv",
)
