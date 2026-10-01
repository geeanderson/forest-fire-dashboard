import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
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

NODES = [
    {
        "id":         "node-01",
        "label":      "Node 01 — Sagres, PT",
        "city":       "Sagres, PT",
        # ML training reference (BR-train/PT-field split, see project-brasil-portugal-rationale)
        "ml_station": "INMET A835 (Maringá, PR)",
        "lat":        37.026837,
        "lon":        -8.956227,
    },
    {
        "id":         "node-02",
        "label":      "Node 02 — Sagres, PT",
        "city":       "Sagres, PT",
        # ML training reference (BR-train/PT-field split, see project-brasil-portugal-rationale)
        "ml_station": "INMET A843 (Dois Vizinhos, PR)",
        "lat":        37.051917,
        "lon":        -8.959559,
    },
    # node-03 removido do dashboard: teste de campo atual usa apenas node-01/node-02
    # (node-03 desligado por colisão de rádio com node-02, ver project-colisao-lora-node02-node03)
]

NODE_IDS = [n["id"] for n in NODES]

RISK_LABELS = {
    0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High", 5: "Critical"
}
RISK_COLORS = {
    0: "#2e7d32", 1: "#558b2f", 2: "#f9a825", 3: "#ef6c00", 4: "#c62828", 5: "#4a148c"
}

PRESETS = {
    "Last 15 min":  timedelta(minutes=15),
    "Last 1 h":     timedelta(hours=1),
    "Last 6 h":     timedelta(hours=6),
    "Last 24 h":    timedelta(hours=24),
    "Last 7 d":     timedelta(days=7),
    "Last 30 d":    timedelta(days=30),
    "Last 90 d":    timedelta(days=90),
    "Last 365 d":   timedelta(days=365),
    "Custom":       None,
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
    key_id    = _secret("AWS_ACCESS_KEY_ID")
    secret_key = _secret("AWS_SECRET_ACCESS_KEY")
    if key_id and secret_key:
        kwargs["aws_access_key_id"]     = key_id
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.resource("dynamodb", **kwargs).Table(DYNAMODB_TABLE)


def load_readings(start: datetime, end: datetime) -> pd.DataFrame:
    table = get_table()
    since = start.isoformat()
    until = end.isoformat()
    frames = []

    for node_id in NODE_IDS:
        items = []
        resp = table.query(
            KeyConditionExpression=(
                Key("device_id").eq(node_id)
                & Key("timestamp").between(since, until)
            ),
            ScanIndexForward=True,
        )
        items.extend(resp["Items"])
        while "LastEvaluatedKey" in resp:
            resp = table.query(
                KeyConditionExpression=(
                    Key("device_id").eq(node_id)
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
st.caption(
    f"Real-time environmental monitoring for wildfire prevention · "
    f"DynamoDB: `{DYNAMODB_TABLE}` · Region: `{AWS_REGION}`"
)

# ---------------------------------------------------------------------------
# Time range picker
# ---------------------------------------------------------------------------

col_preset, col_refresh = st.columns([5, 1])
with col_preset:
    preset = st.radio(
        "Time range",
        options=list(PRESETS.keys()),
        index=4,          # Last 7d as default (covers June data)
        horizontal=True,
        label_visibility="collapsed",
    )
with col_refresh:
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()

now_utc = datetime.now(timezone.utc)

if PRESETS[preset] is not None:
    start_dt = now_utc - PRESETS[preset]
    end_dt   = now_utc
else:
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start date", value=(now_utc - timedelta(days=30)).date())
        start_time = st.time_input("Start time", value=datetime.min.time())
    with col_e:
        end_date = st.date_input("End date", value=now_utc.date())
        end_time = st.time_input("End time", value=datetime.max.time().replace(second=0, microsecond=0))
    start_dt = datetime.combine(start_date, start_time, tzinfo=timezone.utc)
    end_dt   = datetime.combine(end_date,   end_time,   tzinfo=timezone.utc)

st.caption(
    f"📅 **{start_dt.strftime('%Y-%m-%d %H:%M UTC')}** → **{end_dt.strftime('%Y-%m-%d %H:%M UTC')}**"
)

df = load_readings(start_dt, end_dt)

if df.empty:
    st.warning(
        f"No readings found between **{start_dt.strftime('%Y-%m-%d %H:%M UTC')}** "
        f"and **{end_dt.strftime('%Y-%m-%d %H:%M UTC')}**."
    )
    st.info("Tip: use **Custom** range to browse historical data (e.g. June 2026).")
    st.stop()

st.caption(
    f"**{len(df)} readings** · "
    f"{df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} → "
    f"{df['timestamp'].max().strftime('%Y-%m-%d %H:%M')} UTC"
)

st.divider()

# ---------------------------------------------------------------------------
# Node cards
# ---------------------------------------------------------------------------

st.subheader("Latest Reading per Node")
cols = st.columns(len(NODES))

for i, node in enumerate(NODES):
    node_df = df[df["device_id"] == node["id"]]
    maps_url = f"https://www.google.com/maps?q={node['lat']},{node['lon']}"

    with cols[i]:
        # Node header with location
        st.markdown(f"### {node['label']}")
        st.markdown(
            f"📍 **{node['city']}**  \n"
            f"`{node['lat']:.5f}, {node['lon']:.5f}`"
        )
        st.caption(f"ML training reference: {node['ml_station']}")
        st.link_button("🗺️ View on Google Maps", maps_url, use_container_width=True)

        if node_df.empty:
            st.info("No data in selected period.")
            continue

        last = node_df.iloc[-1]
        risk  = int(last.get("fire_risk", -1)) if pd.notna(last.get("fire_risk")) else -1
        label = RISK_LABELS.get(risk, "N/A")
        color = RISK_COLORS.get(risk, "#555")
        prob  = last.get("fire_probability")
        prob_str = f"{float(prob)*100:.1f}%" if pd.notna(prob) else "N/A"

        st.caption(f"Last update: `{last['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}`")

        c1, c2 = st.columns(2)
        c1.metric("Temperature (°C)", f"{last.get('temperature', 0):.1f}")
        c2.metric("Humidity (%)",      f"{last.get('humidity', 0):.1f}")

        c3, c4 = st.columns(2)
        c3.metric("Air Quality",       f"{last.get('air_quality', 0):.0f}")
        days = last.get("days_without_rain")
        c4.metric("Days w/o Rain", int(days) if pd.notna(days) else "N/A")

        st.markdown(
            f"<div style='background:{color};padding:10px;border-radius:8px;"
            f"text-align:center;color:white;font-weight:bold;margin-top:8px;font-size:1.05rem'>"
            f"Fire Risk: {label} ({risk}/5)"
            f"<br><span style='font-size:0.85rem;font-weight:normal'>"
            f"ML Probability: {prob_str}</span></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Node map overview
# ---------------------------------------------------------------------------

with st.expander("🗺️ Node Locations Map", expanded=False):
    map_df = pd.DataFrame([
        {
            "Node":       n["label"],
            "City":       n["city"],
            "ML Station": n["ml_station"],
            "lat":        n["lat"],
            "lon":        n["lon"],
        }
        for n in NODES
    ])
    st.map(map_df, latitude="lat", longitude="lon", size=40, zoom=12)
    st.dataframe(
        map_df[["Node", "City", "ML Station", "lat", "lon"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Time series charts
# ---------------------------------------------------------------------------

st.subheader("Time Series")

NODE_LABELS = {n["id"]: n["label"] for n in NODES}
df["Node"] = df["device_id"].map(NODE_LABELS)

TEMP_COLOR = "#e65100"
HUM_COLOR  = "#1565c0"


def temp_humidity_chart(node_df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    show_temp = "Temperature" in metrics
    show_hum  = "Humidity" in metrics
    both = show_temp and show_hum
    fig = make_subplots(specs=[[{"secondary_y": both}]])
    if show_temp:
        fig.add_trace(
            go.Scatter(x=node_df["timestamp"], y=node_df["temperature"],
                       name="Temperature (°C)", mode="lines",
                       line=dict(color=TEMP_COLOR)),
            secondary_y=False,
        )
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False,
                         title_font_color=TEMP_COLOR)
    if show_hum:
        fig.add_trace(
            go.Scatter(x=node_df["timestamp"], y=node_df["humidity"],
                       name="Humidity (%)", mode="lines",
                       line=dict(color=HUM_COLOR)),
            secondary_y=both,
        )
        fig.update_yaxes(title_text="Humidity (%)", secondary_y=both,
                         title_font_color=HUM_COLOR)
    fig.update_layout(hovermode="x unified",
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


for node in NODES:
    node_df = df[df["device_id"] == node["id"]]
    st.markdown(f"#### {node['label']}")

    if node_df.empty:
        st.info("No data in selected period.")
        continue

    tab_th, tab_risk, tab_prob = st.tabs([
        "🌡️💧 Temperature & Humidity",
        "🔥 Fire Risk (0–5)",
        "📊 ML Probability",
    ])

    with tab_th:
        metrics = st.multiselect(
            "Metrics",
            options=["Temperature", "Humidity"],
            default=["Temperature", "Humidity"],
            key=f"metrics_{node['id']}",
        )
        if metrics:
            st.plotly_chart(temp_humidity_chart(node_df, metrics),
                            use_container_width=True, key=f"th_{node['id']}")
        else:
            st.info("Select at least one metric.")

    with tab_risk:
        if "fire_risk" in node_df.columns:
            fig = px.line(node_df, x="timestamp", y="fire_risk",
                          labels={"fire_risk": "Level (0–5)", "timestamp": ""},
                          range_y=[0, 5])
            fig.add_hrect(y0=3, y1=5, fillcolor="red", opacity=0.08,
                          line_width=0, annotation_text="High risk zone")
            st.plotly_chart(fig, use_container_width=True, key=f"risk_{node['id']}")
            st.caption("Scale: 0=None · 1=Low · 2=Moderate · 3=High · 4=Very High · 5=Critical")
        else:
            st.info("Column fire_risk not found in data.")

    with tab_prob:
        if "fire_probability" in node_df.columns:
            fig = px.line(node_df, x="timestamp", y="fire_probability",
                          labels={"fire_probability": "Probability", "timestamp": ""},
                          range_y=[0, 1])
            fig.add_hrect(y0=0.5, y1=1.0, fillcolor="orange", opacity=0.06,
                          line_width=0, annotation_text="P > 50%")
            fig.add_hrect(y0=0.7, y1=1.0, fillcolor="red", opacity=0.06,
                          line_width=0, annotation_text="P > 70%")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True, key=f"prob_{node['id']}")
            st.caption(
                "Random Forest model · F1=0.748 · AUC=0.801 · "
                "Features: temperature, humidity, days without rain"
            )
        else:
            st.info("Column fire_probability not found in data.")

st.divider()

# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------

st.subheader("Readings Table")

show_cols = [c for c in [
    "Node", "timestamp", "temperature", "humidity", "air_quality",
    "days_without_rain", "fire_risk", "fire_probability", "source", "battery_voltage",
] if c in df.columns]

node_filter = st.multiselect(
    "Node",
    options=[n["label"] for n in NODES],
    default=[n["label"] for n in NODES],
    key="table_node_filter",
)
table_df = df[df["Node"].isin(node_filter)]

st.dataframe(
    table_df[show_cols].sort_values("timestamp", ascending=False),
    use_container_width=True,
    height=320,
)

csv_data = table_df[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Export CSV",
    data=csv_data,
    file_name=f"forest_fire_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("Forest Fire IoT Monitoring System")
