import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Retail Zone Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Design System ────────────────────────────────────────────────────────────
# Cohesive palette + reusable Plotly template so every chart shares one look.

PALETTE = {
    "bg":        "#0e1117",
    "surface":   "#161b26",
    "surface_2": "#1d2330",
    "border":    "#2a3142",
    "text":      "#e6e9ef",
    "muted":     "#9aa4b8",
    "accent":    "#5b8def",
    "accent_2":  "#7c5cff",
    "teal":      "#1fb6a8",
    "amber":     "#f5a524",
    "rose":      "#f3506c",
    "green":     "#2dd4a7",
}

SEQ_BLUE   = ["#1d2942", "#2c4474", "#3a63ad", "#5b8def", "#9bc0ff"]
SEQ_AMBER  = ["#3a2a12", "#6b4d18", "#a87420", "#f5a524", "#ffd27a"]
SEQ_GREEN  = ["#10331f", "#1a6b3e", "#1fb6a8", "#2dd4a7", "#a7f3d0"]
SEQ_PURPLE = ["#241a40", "#3d2c74", "#5d44ad", "#7c5cff", "#b9a6ff"]
SEQ_HEAT   = ["#161b26", "#23314f", "#2c4d6b", "#1fb6a8", "#2dd4a7", "#f5a524", "#f3506c"]
CATEGORICAL = ["#5b8def", "#1fb6a8", "#f5a524", "#7c5cff", "#f3506c", "#2dd4a7", "#9bc0ff", "#ffd27a"]

retail_template = go.layout.Template()
retail_template.layout = go.Layout(
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=PALETTE["text"], size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(size=17, color=PALETTE["text"]), x=0.01, xanchor="left"),
    colorway=CATEGORICAL,
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["muted"]),
        title_font=dict(color=PALETTE["muted"]),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["muted"]),
        title_font=dict(color=PALETTE["muted"]),
    ),
    legend=dict(font=dict(color=PALETTE["muted"]), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=20, r=20, t=60, b=20),
    hoverlabel=dict(bgcolor=PALETTE["surface_2"], font=dict(color=PALETTE["text"])),
)
pio.templates["retail"] = retail_template
pio.templates.default = "retail"


def style_chart(fig, height=380):
    """Apply the shared template + sizing to any Plotly figure."""
    fig.update_layout(template="retail", height=height)
    return fig


# ─── Theme CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

    .stApp {
        background:
            radial-gradient(1200px 600px at 12% -8%, rgba(91,141,239,0.10), transparent 60%),
            radial-gradient(1000px 600px at 95% 0%, rgba(124,92,255,0.08), transparent 55%),
            #0e1117;
        color: #e6e9ef;
    }

    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }

    /* Hero header */
    .rz-hero {
        background: linear-gradient(135deg, rgba(91,141,239,0.16), rgba(124,92,255,0.10));
        border: 1px solid #2a3142;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 40px -18px rgba(0,0,0,0.7);
    }
    .rz-hero h1 {
        font-size: 1.85rem; font-weight: 700; margin: 0;
        color: #f4f6fb; letter-spacing: -0.4px;
    }
    .rz-hero p { margin: 0.35rem 0 0 0; color: #9aa4b8; font-size: 0.96rem; }
    .rz-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(45,212,167,0.12); color: #2dd4a7;
        border: 1px solid rgba(45,212,167,0.30);
        padding: 3px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        margin-top: 0.7rem;
    }

    /* Section headers */
    .rz-section {
        display: flex; align-items: center; gap: 10px;
        margin: 0.4rem 0 0.2rem 0;
    }
    .rz-section .bar {
        width: 4px; height: 22px; border-radius: 4px;
        background: linear-gradient(180deg, #5b8def, #7c5cff);
    }
    .rz-section h2 {
        font-size: 1.25rem; font-weight: 650; margin: 0; color: #eef1f7;
        letter-spacing: -0.2px;
    }
    .rz-sub { color: #9aa4b8; font-size: 0.88rem; margin: 0.1rem 0 0.9rem 14px; }

    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #1a2030, #161b26);
        border: 1px solid #2a3142;
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 8px 30px -20px rgba(0,0,0,0.9);
        transition: transform .15s ease, border-color .15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #3a63ad;
    }
    div[data-testid="stMetricLabel"] p {
        color: #9aa4b8 !important; font-size: 0.82rem !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #f4f6fb !important; font-weight: 700 !important; font-size: 1.9rem !important;
    }

    /* Cards / charts */
    div[data-testid="stPlotlyChart"], .stDataFrame {
        background: #161b26;
        border: 1px solid #2a3142;
        border-radius: 16px;
        padding: 6px;
        box-shadow: 0 8px 30px -22px rgba(0,0,0,0.9);
    }

    hr { border-color: #2a3142 !important; }

    /* Alert cards */
    .rz-alert {
        border-radius: 14px; padding: 0.85rem 1.05rem; margin-bottom: 0.7rem;
        border: 1px solid; display: flex; gap: 12px; align-items: flex-start;
        font-size: 0.92rem; line-height: 1.45;
    }
    .rz-alert .ic { font-size: 1.15rem; line-height: 1.3; }
    .rz-alert.crit { background: rgba(243,80,108,0.10); border-color: rgba(243,80,108,0.40); color: #ffd2da; }
    .rz-alert.warn { background: rgba(245,165,36,0.10); border-color: rgba(245,165,36,0.38); color: #ffe6bd; }
    .rz-alert.info { background: rgba(91,141,239,0.10); border-color: rgba(91,141,239,0.38); color: #cfe0ff; }
    .rz-alert.ok   { background: rgba(45,212,167,0.10); border-color: rgba(45,212,167,0.38); color: #c5f5e6; }
    .rz-alert b { color: #ffffff; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #11151f; border-right: 1px solid #2a3142;
    }
    section[data-testid="stSidebar"] h3 { color: #eef1f7; }

    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #3a63ad, #5b8def);
        color: #fff; border: 0; border-radius: 10px; font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stDownloadButton button:hover { filter: brightness(1.08); }

    div[data-testid="stExpander"] {
        border: 1px solid #2a3142; border-radius: 14px; background: #161b26;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title, subtitle=None):
    """Render a consistent section header (presentation only)."""
    st.markdown(
        f'<div class="rz-section"><div class="bar"></div><h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<div class="rz-sub">{subtitle}</div>', unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────

DB_PATH = r"C:\Retail-Analytics\Retail-Analytics\dwell_events.db"

@st.cache_data(ttl=30)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM dwell_events", conn)
    conn.close()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"]  = pd.to_datetime(df["exit_time"])
    df["hour"]       = df["entry_time"].dt.hour
    df["minute"]     = df["entry_time"].dt.minute
    return df

df = load_data()

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="rz-hero">
        <h1>🛒 Retail Zone Analytics</h1>
        <p>Store performance insights powered by computer vision</p>
        <span class="rz-pill">● Live · auto-refresh every 30s</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.error("No data found in database. Run main.py first.")
    st.stop()

# ─── KPI Row ──────────────────────────────────────────────────────────────────

section("Store Overview", "Headline metrics across the current tracking session")

total_visitors = df["track_id"].nunique()
total_events   = len(df)
avg_dwell      = round(df["dwell_seconds"].mean(), 1)
busiest_zone   = df.groupby("zone")["track_id"].nunique().idxmax()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Unique Visitors", total_visitors)
with col2:
    st.metric("Total Zone Events", total_events)
with col3:
    st.metric("Avg Dwell Time", f"{avg_dwell}s")
with col4:
    st.metric("Most Visited Zone", busiest_zone)

st.divider()

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
# Interactive widgets that filter/select over the SAME data (no recompute of metrics).

with st.sidebar:
    st.markdown("### 🎛️ Controls")
    st.caption("Drill into a single zone or export the raw event feed.")

    zone_options = ["All Zones"] + sorted(df["zone"].unique().tolist())
    selected_zone = st.selectbox("Zone drill-down", zone_options, index=0)

    st.markdown("---")
    st.markdown("### 📤 Export")
    events_csv = df[["track_id", "zone", "entry_time", "exit_time", "dwell_seconds"]].to_csv(index=False)
    st.download_button(
        "Download events (CSV)",
        data=events_csv,
        file_name="dwell_events.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Drill-down view: a filtered VIEW of already-loaded data (does not affect any metric above/below).
if selected_zone != "All Zones":
    zdf = df[df["zone"] == selected_zone]
    section(f"Zone Drill-down · {selected_zone}", "Filtered snapshot for the selected zone")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.metric("Unique Visitors (zone)", zdf["track_id"].nunique())
    with dcol2:
        st.metric("Zone Events (zone)", len(zdf))
    with dcol3:
        st.metric("Avg Dwell (zone)", f"{round(zdf['dwell_seconds'].mean(), 1)}s")
    st.divider()

# ─── Zone Performance ─────────────────────────────────────────────────────────

section("Zone Performance", "Footfall and dwell behaviour, side by side")

col_left, col_right = st.columns(2)

with col_left:
    zone_visitors = df.groupby("zone")["track_id"].nunique().reset_index()
    zone_visitors.columns = ["Zone", "Unique Visitors"]
    zone_visitors = zone_visitors.sort_values("Unique Visitors", ascending=False)

    fig1 = px.bar(
        zone_visitors,
        x="Zone",
        y="Unique Visitors",
        title="Unique Visitors Per Zone",
        color="Unique Visitors",
        color_continuous_scale=SEQ_BLUE,
        text="Unique Visitors"
    )
    fig1.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
    fig1.update_layout(showlegend=False, coloraxis_showscale=False)
    style_chart(fig1)
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    zone_dwell = df.groupby("zone")["dwell_seconds"].mean().reset_index()
    zone_dwell.columns = ["Zone", "Avg Dwell (seconds)"]
    zone_dwell["Avg Dwell (seconds)"] = zone_dwell["Avg Dwell (seconds)"].round(1)
    zone_dwell = zone_dwell.sort_values("Avg Dwell (seconds)", ascending=False)

    fig2 = px.bar(
        zone_dwell,
        x="Zone",
        y="Avg Dwell (seconds)",
        title="Average Dwell Time Per Zone (seconds)",
        color="Avg Dwell (seconds)",
        color_continuous_scale=SEQ_AMBER,
        text="Avg Dwell (seconds)"
    )
    fig2.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
    fig2.update_layout(showlegend=False, coloraxis_showscale=False)
    style_chart(fig2)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─── Zone Engagement Score ────────────────────────────────────────────────────

section(
    "Zone Engagement Score",
    "Combines visitor count and dwell time to rank which zones are truly engaging customers",
)

zone_score = df.groupby("zone").agg(
    visitors=("track_id", "nunique"),
    avg_dwell=("dwell_seconds", "mean")
).reset_index()

zone_score["visitor_score"] = (zone_score["visitors"] / zone_score["visitors"].max()) * 100
zone_score["dwell_score"]   = (zone_score["avg_dwell"] / zone_score["avg_dwell"].max()) * 100
zone_score["engagement"]    = ((zone_score["visitor_score"] + zone_score["dwell_score"]) / 2).round(1)
zone_score                  = zone_score.sort_values("engagement", ascending=False)

fig3 = px.bar(
    zone_score,
    x="zone",
    y="engagement",
    title="Zone Engagement Score (0-100)",
    color="engagement",
    color_continuous_scale=SEQ_GREEN,
    text="engagement"
)
fig3.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
fig3.update_layout(showlegend=False, coloraxis_showscale=False)
style_chart(fig3, height=400)
st.plotly_chart(fig3, use_container_width=True)

# ─── Zone Engagement / Dwell Heatmap ──────────────────────────────────────────
# Built purely from per-zone metrics already computed in zone_score (no new computation).

section(
    "Engagement & Dwell Heatmap",
    "Normalised per-zone metrics on one matrix for quick visual comparison",
)

heat_df = zone_score.sort_values("engagement", ascending=False)
heat_matrix = [
    heat_df["visitor_score"].round(1).tolist(),
    heat_df["dwell_score"].round(1).tolist(),
    heat_df["engagement"].tolist(),
]
heat_rows = ["Visitor Score", "Dwell Score", "Engagement"]

fig_heat = go.Figure(
    data=go.Heatmap(
        z=heat_matrix,
        x=heat_df["zone"].tolist(),
        y=heat_rows,
        colorscale=SEQ_HEAT,
        text=heat_matrix,
        texttemplate="%{text}",
        textfont=dict(color="#0e1117", size=12),
        hovertemplate="Zone: %{x}<br>%{y}: %{z}<extra></extra>",
        colorbar=dict(title="Score", tickfont=dict(color=PALETTE["muted"])),
    )
)
fig_heat.update_layout(title="Per-Zone Metric Heatmap (0-100)")
style_chart(fig_heat, height=320)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ─── Zone Visit Frequency ─────────────────────────────────────────────────────

section("Zone Visit Frequency", "How many times each zone was entered during the session")

zone_freq = df.groupby("zone").size().reset_index(name="Total Visits")
zone_freq = zone_freq.sort_values("Total Visits", ascending=False)

fig4 = px.bar(
    zone_freq,
    x="zone",
    y="Total Visits",
    title="Total Zone Entries",
    color="Total Visits",
    color_continuous_scale=SEQ_PURPLE,
    text="Total Visits"
)
fig4.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
fig4.update_layout(showlegend=False, coloraxis_showscale=False)
style_chart(fig4)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ─── Dwell Time Ranked Table ──────────────────────────────────────────────────

section("Zone Dwell Time Breakdown", "Ranked by average time customers spend in each zone")

dwell_table = df.groupby("zone").agg(
    Total_Visits=("track_id", "count"),
    Avg_Dwell=("dwell_seconds", "mean"),
    Max_Dwell=("dwell_seconds", "max"),
    Min_Dwell=("dwell_seconds", "min")
).reset_index()

dwell_table["Avg_Dwell"] = dwell_table["Avg_Dwell"].round(1)
dwell_table["Max_Dwell"] = dwell_table["Max_Dwell"].round(1)
dwell_table["Min_Dwell"] = dwell_table["Min_Dwell"].round(1)
dwell_table.columns      = ["Zone", "Total Visits", "Avg Dwell (s)", "Max Dwell (s)", "Min Dwell (s)"]
dwell_table              = dwell_table.sort_values("Avg Dwell (s)", ascending=False)

st.dataframe(dwell_table, use_container_width=True, hide_index=True)

st.divider()

# ─── Customer Journey ─────────────────────────────────────────────────────────

section("Customer Journey", "Zone-to-zone flow across every tracked customer")

journey = df.sort_values(["track_id", "entry_time"])
journey_summary = journey.groupby("track_id")["zone"].apply(
    lambda x: " → ".join(x.tolist())
).reset_index()
journey_summary.columns = ["Customer ID", "Zone Journey"]

# Build a Sankey of zone→zone transitions from the SAME ordered journey data.
# (journey is already sorted by track_id, entry_time above.)
journey_seq = journey.groupby("track_id")["zone"].apply(list)
flow_counts = {}
for path in journey_seq:
    for a, b in zip(path[:-1], path[1:]):
        flow_counts[(a, b)] = flow_counts.get((a, b), 0) + 1

if flow_counts:
    sankey_nodes = sorted({z for pair in flow_counts for z in pair})
    node_index = {z: i for i, z in enumerate(sankey_nodes)}
    node_colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(sankey_nodes))]

    link_source = [node_index[a] for (a, b) in flow_counts]
    link_target = [node_index[b] for (a, b) in flow_counts]
    link_value  = [v for v in flow_counts.values()]

    def _rgba(hex_color, alpha=0.45):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [_rgba(node_colors[s]) for s in link_source]

    fig_sankey = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(
                label=sankey_nodes,
                color=node_colors,
                pad=18,
                thickness=18,
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            link=dict(
                source=link_source,
                target=link_target,
                value=link_value,
                color=link_colors,
                hovertemplate="%{source.label} → %{target.label}<br>Transitions: %{value}<extra></extra>",
            ),
        )]
    )
    fig_sankey.update_layout(title="Zone-to-Zone Flow (Sankey)")
    style_chart(fig_sankey, height=460)
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.info("Not enough sequential movement to render a zone-flow diagram yet.")

with st.expander("🧭 Per-customer zone paths"):
    st.dataframe(
        journey_summary,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# ─── Manager Alerts ───────────────────────────────────────────────────────────

section("Manager Alerts", "Prioritised operational signals from this session")

EXCLUDE_FROM_ALERTS = {"Entrance"}

alerts = []

filtered_score = zone_score[~zone_score["zone"].isin(EXCLUDE_FROM_ALERTS)]
filtered_dwell = zone_dwell[~zone_dwell["Zone"].isin(EXCLUDE_FROM_ALERTS)]

# Low engagement zone
if not filtered_score.empty:
    low_zone = filtered_score.iloc[-1]
    if low_zone["engagement"] < 50:
        alerts.append(
            ("warn",
             f"**{low_zone['zone']}** has the lowest engagement score "
             f"({low_zone['engagement']}/100). Consider repositioning products or improving signage.")
        )

# Zone with very short dwell
if not filtered_dwell.empty:
    short_dwell_zone = filtered_dwell.sort_values("Avg Dwell (seconds)").iloc[0]
    if short_dwell_zone["Avg Dwell (seconds)"] < 8:
        alerts.append(
            ("warn",
             f"**{short_dwell_zone['Zone']}** average dwell time is only "
             f"{short_dwell_zone['Avg Dwell (seconds)']}s. "
             f"Customers are passing through without engaging.")
        )

# Dead zones excluding entrance
all_zones     = {"Checkout Counter", "Center Aisle", "Back Aisle", "Right Aisle"}
visited_zones = set(df["zone"].unique())
dead_zones    = all_zones - visited_zones
for z in dead_zones:
    alerts.append(("crit", f"**{z}** recorded zero visitor activity during this session."))

# Checkout bottleneck
checkout_data = df[df["zone"] == "Checkout Counter"]["dwell_seconds"]
if not checkout_data.empty:
    checkout_avg = checkout_data.mean()
    if checkout_avg > 60:
        alerts.append(
            ("crit",
             f"**Checkout Counter** average wait is {round(checkout_avg, 1)}s. "
             f"Customers may be experiencing long queue times.")
        )

# Presentation only: render each alert as a severity-tiered card with an icon/colour.
SEVERITY_META = {
    "crit": ("🔴", "crit"),
    "warn": ("🟠", "warn"),
    "info": ("🔵", "info"),
}


def _md_bold_to_html(text):
    parts = text.split("**")
    out = ""
    for i, p in enumerate(parts):
        out += f"<b>{p}</b>" if i % 2 == 1 else p
    return out


if alerts:
    for severity, message in alerts:
        icon, css = SEVERITY_META.get(severity, SEVERITY_META["warn"])
        st.markdown(
            f'<div class="rz-alert {css}"><span class="ic">{icon}</span>'
            f'<span>{_md_bold_to_html(message)}</span></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="rz-alert ok"><span class="ic">🟢</span>'
        '<span>All zones are performing within normal range.</span></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ─── Raw Data ─────────────────────────────────────────────────────────────────

with st.expander("📋 Raw Event Log"):
    st.dataframe(
        df[["track_id", "zone", "entry_time", "exit_time", "dwell_seconds"]]
        .sort_values("dwell_seconds", ascending=False),
        use_container_width=True,
        hide_index=True
    )