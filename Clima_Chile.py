import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

st.set_page_config(page_title="Clima_Chile", page_icon="⛅", layout="wide")

# -----------------------------
# Utilidades
# -----------------------------
CITIES ={
    "Santiago": (-33.45, -70.66),
    "Valparaíso": (-33.05, -71.62),
    "Concepción": (-36.83, -73.05),
    "Antofagasta": (-23.65, -70.40),
    "Coyhaique": (-45.58, -72.07),}


DEFAULT_END = date.today() - timedelta(days=1)
DEFAULT_START = DEFAULT_END - timedelta(days=365)

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max"
]

VAR_LABELS = {
    "temperature_2m_max": "T° Máx (°C)",
    "temperature_2m_min": "T° Mín (°C)",
    "precipitation_sum": "Precipitación (mm)",
    "windspeed_10m_max": "Viento máx (km/h)"
}

@st.cache_data(show_spinner=False)
def fetch_openmeteo_daily(lat, lon, start, end):
    """
    Descarga datos diarios de Open-Meteo (API de archivo / histórico).
    Retorna DataFrame con columnas: date, temperature_2m_max, temperature_2m_min, precipitation_sum, windspeed_10m_max
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "windspeed_unit": "kmh"
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "daily" not in data:
        return pd.DataFrame()
    daily = data["daily"]

    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for k in DAILY_VARS:
        df[k] = daily.get(k, [np.nan] * len(df))
    return df.sort_values("date").reset_index(drop=True)

def compute_kpis(df):
    kpis = {}
    if df.empty:
        return {"días": 0, "T° media": np.nan, "Total precip": np.nan, "Día más caluroso": None}

    temp_mean = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
    kpis["días"] = len(df)
    kpis["T° media"] = round(temp_mean.mean(), 1)
    kpis["Total precip"] = round(df["precipitation_sum"].sum(), 1)

    hottest_idx = df["temperature_2m_max"].idxmax()
    hottest_day = df.loc[hottest_idx, "date"].date() if pd.notna(hottest_idx) else None
    kpis["Día más caluroso"] = hottest_day
    return kpis

def monthly_summary(df):
    if df.empty:
        return df
    m = df.copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    agg = m.groupby("month").agg(
        Tmax=("temperature_2m_max", "mean"),
        Tmin=("temperature_2m_min", "mean"),
        PrecipTotal=("precipitation_sum", "sum"),
        VientoMax=("windspeed_10m_max", "mean")
    ).reset_index()
    return agg

def download_button_csv(df, filename="clima_chile.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv, file_name=filename, mime="text/csv")

# -----------------------------
# UI – Sidebar
# -----------------------------
st.title("⛅ Clima en Chile – Últimos 12 meses")
st.caption("Datos diarios desde Open-Meteo (histórico). Exporta, filtra y visualiza.")

with st.sidebar:
    st.header("Parámetros")
    city = st.selectbox("Ciudad", list(CITIES.keys()), index=0)
    dr = st.date_input("Rango de fechas (máx. 1 año aprox.)",
                       (DEFAULT_START, DEFAULT_END),
                       min_value=date(1979, 1, 1),
                       max_value=DEFAULT_END)
    if isinstance(dr, tuple):
        start_date, end_date = dr
    else:
         start_date, end_date = DEFAULT_START, DEFAULT_END

    # Asegurar rango válido
    if start_date > end_date:
        st.error("El inicio debe ser anterior al fin.")
    selected_vars = st.multiselect(
        "Variables a visualizar",
        options=[VAR_LABELS[v] for v in DAILY_VARS],
        default=[VAR_LABELS["temperature_2m_max"], VAR_LABELS["temperature_2m_min"], VAR_LABELS["precipitation_sum"]],
    )
    # Invertir el diccionario de etiquetas para mapear nombre mostrado -> key
    inv_labels = {v: k for k, v in VAR_LABELS.items()}
    vars_keys = [inv_labels[v] for v in selected_vars] if selected_vars else DAILY_VARS

# -----------------------------
# Datos
# -----------------------------
lat, lon = CITIES[city]
with st.spinner(f"Descargando datos para {city} ({start_date} → {end_date})..."):
    df = fetch_openmeteo_daily(lat, lon, start_date, end_date)

if df.empty:
    st.warning("No se recibieron datos para el rango seleccionado.")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
k = compute_kpis(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Días", k["días"])
c2.metric("T° media (°C)", k["T° media"])
c3.metric("Precipitación total (mm)", k["Total precip"])
# Suponiendo que k["Día más caluroso"] es un objeto datetime.date
caloroso_date = k["Día más caluroso"].strftime('%Y-%m-%d')  # Convierte a string en formato 'YYYY-MM-DD'
c4.metric("Día más caluroso", caloroso_date)



st.divider()

# -----------------------------
# Tabla + descarga
# -----------------------------
st.subheader(f"Datos diarios – {city}")
st.dataframe(df, use_container_width=True, height=300)
download_button_csv(df, f"clima_{city.lower().replace(' ', '')}{start_date}_{end_date}.csv")

st.divider()

# -----------------------------
# Visualizaciones (Altair / Matplotlib)
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Evolución temporal")
    # Linea para cada variable seleccionada
    base = alt.Chart(df).encode(x=alt.X("date:T", title="Fecha"))
    layers = []
    palette = {"temperature_2m_max": "#e45756",
               "temperature_2m_min": "#4ca2ff",
               "precipitation_sum": "#6cc24a",
               "windspeed_10m_max": "#FFA500"}
    for v in vars_keys:
        layers.append(
            base.mark_line(point=False, color=palette.get(v, "#999")).encode(
                y=alt.Y(f"{v}:Q", title=VAR_LABELS[v]),
                tooltip=[alt.Tooltip("date:T", title="Fecha"),
                         alt.Tooltip(f"{v}:Q", title=VAR_LABELS[v], format=".2f")]
            ).interactive()
        )
    chart = alt.layer(*layers).resolve_scale(y="independent")
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("Histogramas")
    num_col = st.selectbox(
        "Variable",
        df.select_dtypes(include="number").columns.map(lambda c: VAR_LABELS.get(c, c)),
    )
    # map label back to key for plotting
    num_key = {v: k for k, v in VAR_LABELS.items()}.get(num_col, num_col)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df[num_key].dropna(), bins=20, edgecolor='white')
    ax.set_title(f"Histograma: {VAR_LABELS.get(num_key, num_key)}")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    st.divider()

# -----------------------------
# Resumen mensual
# -----------------------------
st.subheader("Resumen mensual")
m = monthly_summary(df)
st.dataframe(m, use_container_width=True)

chart_m = alt.Chart(m).encode(x=alt.X("month:T", title="Mes"))
l1 = chart_m.mark_line(color="#e45756").encode(y=alt.Y("Tmax:Q", title="Tmax/Tmin"))
l2 = chart_m.mark_line(color="#4ca2ff").encode(y="Tmin:Q")
b1 = chart_m.mark_bar(color="#6cc24a").encode(y=alt.Y("PrecipTotal:Q", title="Precipitación (mm)"))

st.altair_chart(alt.layer(b1, l1, l2).resolve_scale(y="independent"), use_container_width=True)

st.caption("Fuente de datos: Open-Meteo Archive API")

