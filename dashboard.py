import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Retail Zone Analytics",
    page_icon="🏪",
    layout="wide"
)

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

st.title("🏪 Retail Zone Analytics Dashboard")
st.caption("Store performance insights powered by computer vision")

if df.empty:
    st.error("No data found in database. Run main.py first.")
    st.stop()

# ─── KPI Row ──────────────────────────────────────────────────────────────────

st.markdown("## Store Overview")

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

# ─── Zone Performance ─────────────────────────────────────────────────────────

st.markdown("## Zone Performance")

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
        color_continuous_scale="Blues",
        text="Unique Visitors"
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(showlegend=False, coloraxis_showscale=False)
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
        color_continuous_scale="Oranges",
        text="Avg Dwell (seconds)"
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─── Zone Engagement Score ────────────────────────────────────────────────────

st.markdown("## Zone Engagement Score")
st.caption("Combines visitor count and dwell time to rank which zones are truly engaging customers")

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
    color_continuous_scale="RdYlGn",
    text="engagement"
)
fig3.update_traces(textposition="outside")
fig3.update_layout(showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ─── Zone Visit Frequency ─────────────────────────────────────────────────────

st.markdown("## Zone Visit Frequency")
st.caption("How many times each zone was entered during the session")

zone_freq = df.groupby("zone").size().reset_index(name="Total Visits")
zone_freq = zone_freq.sort_values("Total Visits", ascending=False)

fig4 = px.bar(
    zone_freq,
    x="zone",
    y="Total Visits",
    title="Total Zone Entries",
    color="Total Visits",
    color_continuous_scale="Purples",
    text="Total Visits"
)
fig4.update_traces(textposition="outside")
fig4.update_layout(showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ─── Dwell Time Ranked Table ──────────────────────────────────────────────────

st.markdown("## Zone Dwell Time Breakdown")
st.caption("Ranked by average time customers spend in each zone")

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

st.markdown("## Customer Journey")
st.caption("Zone path each tracked customer took through the store")

journey = df.sort_values(["track_id", "entry_time"])
journey_summary = journey.groupby("track_id")["zone"].apply(
    lambda x: " → ".join(x.tolist())
).reset_index()
journey_summary.columns = ["Customer ID", "Zone Journey"]

st.dataframe(
    journey_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ─── Manager Alerts ───────────────────────────────────────────────────────────

st.markdown("## ⚠️ Manager Alerts")

EXCLUDE_FROM_ALERTS = {"Entrance"}

alerts = []

filtered_score = zone_score[~zone_score["zone"].isin(EXCLUDE_FROM_ALERTS)]
filtered_dwell = zone_dwell[~zone_dwell["Zone"].isin(EXCLUDE_FROM_ALERTS)]

# Low engagement zone
if not filtered_score.empty:
    low_zone = filtered_score.iloc[-1]
    if low_zone["engagement"] < 50:
        alerts.append(
            f"**{low_zone['zone']}** has the lowest engagement score "
            f"({low_zone['engagement']}/100). Consider repositioning products or improving signage."
        )

# Zone with very short dwell
if not filtered_dwell.empty:
    short_dwell_zone = filtered_dwell.sort_values("Avg Dwell (seconds)").iloc[0]
    if short_dwell_zone["Avg Dwell (seconds)"] < 8:
        alerts.append(
            f"**{short_dwell_zone['Zone']}** average dwell time is only "
            f"{short_dwell_zone['Avg Dwell (seconds)']}s. "
            f"Customers are passing through without engaging."
        )

# Dead zones excluding entrance
all_zones     = {"Checkout Counter", "Center Aisle", "Back Aisle", "Right Aisle"}
visited_zones = set(df["zone"].unique())
dead_zones    = all_zones - visited_zones
for z in dead_zones:
    alerts.append(f"**{z}** recorded zero visitor activity during this session.")

# Checkout bottleneck
checkout_data = df[df["zone"] == "Checkout Counter"]["dwell_seconds"]
if not checkout_data.empty:
    checkout_avg = checkout_data.mean()
    if checkout_avg > 60:
        alerts.append(
            f"**Checkout Counter** average wait is {round(checkout_avg, 1)}s. "
            f"Customers may be experiencing long queue times."
        )

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("All zones are performing within normal range.")

st.divider()

# ─── Raw Data ─────────────────────────────────────────────────────────────────

with st.expander("📋 Raw Event Log"):
    st.dataframe(
        df[["track_id", "zone", "entry_time", "exit_time", "dwell_seconds"]]
        .sort_values("dwell_seconds", ascending=False),
        use_container_width=True,
        hide_index=True
    )