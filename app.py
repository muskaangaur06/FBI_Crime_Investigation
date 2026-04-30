import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Crime Forecast · Vancouver",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.path.dirname(os.path.abspath(__file__))

PALETTE = {
    "red"   : "#e74c3c",
    "blue"  : "#3498db",
    "dark"  : "#0f1117",
    "card"  : "#1a1d27",
    "border": "#2d3047",
    "muted" : "#8892a4",
    "white" : "#fafafa",
}

MONTH_MAP = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

st.markdown("""
<style>
    footer, header { visibility: hidden; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2d3047;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8892a4 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #fafafa !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
    .section-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8892a4;
        margin: 24px 0 8px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d3047;
    }
    .tag {
        display: inline-block;
        background: #2d3047;
        color: #fafafa;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin-right: 4px;
    }
    .chat-user {
        background: #2d3047;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
    }
    .chat-bot {
        background: #1f2535;
        border-left: 3px solid #e74c3c;
        border-radius: 2px 12px 12px 12px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
    }
    [data-testid="stSidebar"] {
        background: #1a1d27;
        border-right: 1px solid #2d3047;
    }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model  = joblib.load(os.path.join(BASE, "xgb_final_model.joblib"))
    scaler = joblib.load(os.path.join(BASE, "scaler.joblib"))
    le     = joblib.load(os.path.join(BASE, "label_encoder.joblib"))
    return model, scaler, le

@st.cache_data(show_spinner=False)
def load_data():
    hist      = pd.read_csv(os.path.join(BASE, "monthly_historical.csv"))
    sub       = pd.read_csv(os.path.join(BASE, "submission.csv"))
    nbhd      = pd.read_csv(os.path.join(BASE, "neighbourhood_crime.csv"))
    hour_type = pd.read_csv(os.path.join(BASE, "hourly_crime_by_type.csv"))
    hour      = hour_type.groupby("HOUR")["Incident_Count"].sum().reset_index()
    hist["Period"] = pd.to_datetime(hist[["YEAR","MONTH"]].assign(DAY=1))
    sub["Period"]  = pd.to_datetime(sub[["YEAR","MONTH"]].assign(DAY=1))
    sub.rename(columns={"Incident_Counts": "Incident_Count"}, inplace=True)
    return hist, sub, nbhd, hour, hour_type

with st.spinner("Loading model..."):
    model, scaler, le       = load_artifacts()
    hist, forecast, nbhd, hour_df, hour_type_df = load_data()

crime_types = sorted(le.classes_.tolist())


st.markdown("## 🔍 Vancouver Crime Forecast Dashboard")
st.markdown('<p style="color:#8892a4;margin-top:-12px;">Predicting monthly crime incidents · Jan 2012 – Jun 2013 · 9 crime types · XGBoost</p>', unsafe_allow_html=True)
st.markdown("---")

# Crime type selector in main area
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    selected_type = st.selectbox("Crime Type", crime_types, index=crime_types.index("Theft from Vehicle"))

ct_hist     = hist[hist["TYPE"] == selected_type]
ct_forecast = forecast[forecast["TYPE"] == selected_type].sort_values("Period")

hist_total  = int(ct_hist["Incident_Count"].sum())
fcast_total = int(ct_forecast["Incident_Count"].sum())
avg_monthly = ct_forecast["Incident_Count"].mean()
peak_row    = ct_forecast.loc[ct_forecast["Incident_Count"].idxmax()]
peak_label  = f"{MONTH_MAP[int(peak_row['MONTH'])]} {int(peak_row['YEAR'])}"
yoy_change  = ((ct_forecast[ct_forecast["YEAR"]==2013]["Incident_Count"].sum() /
                ct_forecast[ct_forecast["YEAR"]==2012]["Incident_Count"].sum()) - 1) * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Historical Total",     f"{hist_total:,}",   "1999 – 2011")
k2.metric("Forecast Total",       f"{fcast_total:,}",  "Jan 2012 – Jun 2013")
k3.metric("Peak Forecast Month",  peak_label,          f"{int(peak_row['Incident_Count'])} incidents")
k4.metric("Avg Monthly Forecast", f"{avg_monthly:.0f}", f"{yoy_change:+.1f}% YoY")

st.markdown("")

st.markdown('<p class="section-title">Historical vs Forecast</p>', unsafe_allow_html=True)

hist_plot = (ct_hist.groupby("Period")["Incident_Count"]
             .sum().reset_index().sort_values("Period"))
hist_plot = hist_plot[hist_plot["Period"] >= "2008-01-01"]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_plot["Period"], y=hist_plot["Incident_Count"],
    mode="lines", name="Historical",
    line=dict(color=PALETTE["blue"], width=2),
    fill="tozeroy", fillcolor="rgba(52,152,219,0.07)"
))
fig.add_trace(go.Scatter(
    x=ct_forecast["Period"], y=ct_forecast["Incident_Count"],
    mode="lines+markers", name="Forecast",
    line=dict(color=PALETTE["red"], width=2.5, dash="dot"),
    marker=dict(size=6, color=PALETTE["red"]),
    fill="tozeroy", fillcolor="rgba(231,76,60,0.07)"
))
fig.add_vrect(
    x0="2012-01-01", x1="2012-02-01",
    fillcolor=PALETTE["muted"], opacity=0.15, line_width=0,
    annotation_text="Forecast Start",
    annotation_font_color=PALETTE["muted"],
    annotation_position="top left"
)
fig.update_layout(
    plot_bgcolor=PALETTE["dark"], paper_bgcolor=PALETTE["dark"],
    font=dict(color=PALETTE["white"], size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor=PALETTE["border"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=PALETTE["border"], showgrid=True, zeroline=False,
               title="Monthly Incidents"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=340, hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown('<p class="section-title">Forecast Breakdown</p>', unsafe_allow_html=True)
    tbl = ct_forecast[["YEAR","MONTH","Incident_Count"]].sort_values(["YEAR","MONTH"]).reset_index(drop=True)
    tbl["Month"] = tbl["MONTH"].map(MONTH_MAP).astype(str) + " " + tbl["YEAR"].astype(str)
    tbl = tbl[["Month","Incident_Count"]].rename(columns={"Incident_Count":"Predicted"})
    st.dataframe(tbl, use_container_width=True, height=340, hide_index=True)

with col_right:
    st.markdown('<p class="section-title">All Crime Types — Forecast Total</p>', unsafe_allow_html=True)
    summary = forecast.groupby("TYPE")["Incident_Count"].sum().reset_index().sort_values("Incident_Count")
    colors  = [PALETTE["red"] if t == selected_type else PALETTE["blue"] for t in summary["TYPE"]]

    fig2 = go.Figure(go.Bar(
        x=summary["Incident_Count"],
        y=[t[:35] for t in summary["TYPE"]],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=summary["Incident_Count"].apply(lambda x: f"{x:,}"),
        textposition="outside", textfont=dict(color=PALETTE["white"], size=11)
    ))
    fig2.update_layout(
        plot_bgcolor=PALETTE["dark"], paper_bgcolor=PALETTE["dark"],
        font=dict(color=PALETTE["white"], size=11),
        xaxis=dict(gridcolor=PALETTE["border"], showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", showgrid=False),
        margin=dict(l=10, r=80, t=10, b=10),
        height=340
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<p class="section-title">Monthly Seasonality Pattern</p>', unsafe_allow_html=True)

col_s1, col_s2 = st.columns(2)

with col_s1:
    monthly_avg = ct_forecast.groupby("MONTH")["Incident_Count"].mean().reset_index()
    bar_colors  = [PALETTE["red"] if m in [6,7,8] else PALETTE["blue"] for m in monthly_avg["MONTH"]]

    fig3 = go.Figure(go.Bar(
        x=[MONTH_MAP[m] for m in monthly_avg["MONTH"]],
        y=monthly_avg["Incident_Count"],
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=monthly_avg["Incident_Count"].round(0).astype(int),
        textposition="outside", textfont=dict(color=PALETTE["white"], size=10)
    ))
    fig3.update_layout(
        title=dict(text="Avg Monthly Forecast (Summer = Red)", font=dict(size=12)),
        plot_bgcolor=PALETTE["dark"], paper_bgcolor=PALETTE["dark"],
        font=dict(color=PALETTE["white"]),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=PALETTE["border"]),
        margin=dict(l=10, r=10, t=40, b=10), height=280
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_s2:
    pivot = (forecast.groupby(["TYPE","MONTH"])["Incident_Count"]
             .mean().reset_index()
             .pivot(index="TYPE", columns="MONTH", values="Incident_Count")
             .fillna(0))
    pivot.columns = [MONTH_MAP[c] for c in pivot.columns]
    pivot.index   = [i[:25] for i in pivot.index]

    fig4 = px.imshow(pivot, color_continuous_scale="Reds",
                     labels=dict(color="Avg Count"), aspect="auto")
    fig4.update_layout(
        title=dict(text="Avg Monthly Forecast Heatmap", font=dict(size=12)),
        plot_bgcolor=PALETTE["dark"], paper_bgcolor=PALETTE["dark"],
        font=dict(color=PALETTE["white"], size=10),
        coloraxis_colorbar=dict(tickfont=dict(color=PALETTE["white"])),
        margin=dict(l=10, r=10, t=40, b=10), height=280
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown('<p class="section-title">Crime Hotspots by Neighbourhood</p>', unsafe_allow_html=True)

fig_geo = px.scatter_mapbox(
    nbhd,
    lat="Latitude", lon="Longitude",
    size="Incident_Count",
    color="Incident_Count",
    color_continuous_scale="Reds",
    hover_name="NEIGHBOURHOOD",
    hover_data={"Incident_Count": True, "Latitude": False, "Longitude": False},
    size_max=55,
    zoom=11,
    center={"lat": 49.258, "lon": -123.108},
    mapbox_style="carto-darkmatter",
    labels={"Incident_Count": "Total Incidents"}
)
fig_geo.update_layout(
    paper_bgcolor=PALETTE["dark"],
    font=dict(color=PALETTE["white"]),
    coloraxis_colorbar=dict(tickfont=dict(color=PALETTE["white"]), title="Incidents"),
    margin=dict(l=0, r=0, t=0, b=0),
    height=440
)
st.plotly_chart(fig_geo, use_container_width=True)

st.markdown('<p class="section-title">Incidents by Hour of Day</p>', unsafe_allow_html=True)

col_h1, col_h2 = st.columns([1.4, 1])

with col_h1:
    hour_colors = [
        PALETTE["red"] if h in range(17, 21) else
        "#e67e22" if h in range(21, 24) or h == 0 else
        PALETTE["blue"]
        for h in hour_df["HOUR"]
    ]
    fig_hour = go.Figure(go.Bar(
        x=hour_df["HOUR"],
        y=hour_df["Incident_Count"],
        marker=dict(color=hour_colors, line=dict(width=0)),
        hovertemplate="Hour %{x}:00 — %{y:,} incidents<extra></extra>"
    ))
    fig_hour.update_layout(
        plot_bgcolor=PALETTE["dark"], paper_bgcolor=PALETTE["dark"],
        font=dict(color=PALETTE["white"], size=11),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", title="Hour of Day",
                   tickvals=list(range(0, 24, 2)),
                   ticktext=[f"{h}:00" for h in range(0, 24, 2)]),
        yaxis=dict(gridcolor=PALETTE["border"], title="Total Incidents"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280
    )
    st.plotly_chart(fig_hour, use_container_width=True)
    st.markdown('<p style="color:#8892a4;font-size:0.75rem;">Red = evening peak (5–8 PM) · Orange = late night</p>',
                unsafe_allow_html=True)

with col_h2:
    ct_hour = hour_type_df[hour_type_df["TYPE"] == selected_type].copy()
    if not ct_hour.empty:
        peak_h = int(ct_hour.loc[ct_hour["Incident_Count"].idxmax(), "HOUR"])
        fig_htype = go.Figure(go.Bar(
            x=ct_hour["HOUR"],
            y=ct_hour["Incident_Count"],
            marker=dict(
                color=[PALETTE["red"] if h == peak_h else PALETTE["blue"] for h in ct_hour["HOUR"]],
                line=dict(width=0)
            ),
            hovertemplate="Hour %{x}:00 — %{y:,}<extra></extra>"
        ))
        fig_htype.update_layout(
            title=dict(text=f"{selected_type[:30]} — by Hour", font=dict(size=11)),
            plot_bgcolor=PALETTE["dark"], paper_bgcolor=PALETTE["dark"],
            font=dict(color=PALETTE["white"], size=10),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", title="Hour",
                       tickvals=list(range(0, 24, 4)),
                       ticktext=[f"{h}:00" for h in range(0, 24, 4)]),
            yaxis=dict(gridcolor=PALETTE["border"]),
            margin=dict(l=10, r=10, t=35, b=10),
            height=280
        )
        st.plotly_chart(fig_htype, use_container_width=True)
        st.markdown(f'<p style="color:#8892a4;font-size:0.75rem;">Peak for this crime type: <b style="color:{PALETTE["red"]}">{peak_h}:00</b></p>',
                    unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<p style="color:#8892a4;font-size:0.72rem;text-align:center;">'
    'FBI Crime Investigation · XGBoost Time Series Forecast · Vancouver 1999–2011</p>',
    unsafe_allow_html=True
)
