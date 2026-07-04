import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="F1 Gap to Car Ahead", layout="centered")

CACHE_DIR = "./f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

CURRENT_YEAR = 2026

# -------------------------
# MAIN
# -------------------------
def compute_gap_by_position(session, driver):
    laps = session.laps
    driver_laps = laps.pick_driver(driver).copy()
    driver_laps = driver_laps.sort_values('LapNumber')

    rows = []

    for _, lap in driver_laps.iterrows():
        lap_num = lap['LapNumber']
        position = lap['Position']
        lap_time_abs = lap['Time']

        if pd.isna(position) or pd.isna(lap_time_abs):
            continue

        if position <= 1:
            gap = 0.0
            ahead_driver = None
        else:
            same_lap = laps[laps['LapNumber'] == lap_num]
            ahead = same_lap[same_lap['Position'] == position - 1]

            if ahead.empty:
                continue

            ahead_time = ahead.iloc[0]['Time']
            gap = (lap_time_abs - ahead_time).total_seconds()
            ahead_driver = ahead.iloc[0]['Driver']

        rows.append({
            "Lap": lap_num,
            "Gap": gap,
            "DriverAhead": ahead_driver
        })

    return pd.DataFrame(rows)


def compute_gap_on_track(session, driver):
    driver_laps = session.laps.pick_driver(driver)

    rows = []
    for _, lap in driver_laps.iterlaps():
        lap_num = lap['LapNumber']
        try:
            car_data = lap.get_car_data().add_distance()
            car_data = car_data.add_driver_ahead()
        except Exception:
            continue

        speed_ms = car_data['Speed'] / 3.6  # km/h -> m/s
        with np.errstate(divide='ignore', invalid='ignore'):
            gap_time = car_data['DistanceToDriverAhead'] / speed_ms
        gap_time = gap_time.replace([np.inf, -np.inf], np.nan).dropna()

        gap_time = gap_time[(gap_time >= 0) & (gap_time < 200)]
        if gap_time.empty:
            continue

        ahead_mode = car_data['DriverAhead'].mode()
        ahead_driver = ahead_mode.iloc[0] if not ahead_mode.empty else None

        rows.append({
            'Lap': lap_num,
            'Gap': gap_time.median(),
            'DriverAhead': ahead_driver,
        })

    return pd.DataFrame(rows)


# -------------------------
# UI
# -------------------------
st.title("🏎️ Gap to Car Ahead Visualizer")

year = st.selectbox("Year", list(range(2018, CURRENT_YEAR + 1))[::-1])

@st.cache_data
def load_schedule(year):
    return fastf1.get_event_schedule(year, include_testing=False)

schedule = load_schedule(year)

event_names = {
    f"Rd {int(row['RoundNumber'])} - {row['EventName']}": int(row['RoundNumber'])
    for _, row in schedule.iterrows()
}

event_label = st.selectbox("Event", list(event_names.keys()))
event_round = event_names[event_label]

session_type = st.selectbox("Session", ["R", "S"])  # Race or Sprint

@st.cache_data
def load_session(year, rnd, session_type):
    session = fastf1.get_session(year, rnd, session_type)
    session.load()
    return session

if "session_obj" not in st.session_state:
    st.session_state.session_obj = None
if "session_key" not in st.session_state:
    st.session_state.session_key = None

current_key = (year, event_round, session_type)

if st.button("Load Session"):
    with st.spinner("Loading session data..."):
        st.session_state.session_obj = load_session(year, event_round, session_type)
        st.session_state.session_key = current_key

if st.session_state.session_obj is not None and st.session_state.session_key == current_key:
    session = st.session_state.session_obj
    drivers = sorted(session.laps['Driver'].unique())
    driver = st.selectbox("Driver", drivers)

    gap_mode = st.radio(
        "Gap mode",
        ["By race position", "Actual car ahead (on-track)"],
        help=(
            "By race position: gap to whoever currently holds the position ahead, "
            "based on race classification.\n\n"
            "Actual car ahead (on-track): gap to whoever is physically ahead on track, "
            "computed from telemetry (speed + distance). Includes lapped cars, "
            "excludes red flags and pit lane. SLOW."
        ),
    )

    if st.button("Generate Plot"):
        if gap_mode == "By race position":
            with st.spinner("Computing gap by race position..."):
                df = compute_gap_by_position(session, driver)
            mode_label = "Gap to Car Ahead (by race position)"
        else:
            with st.spinner("Computing on-track gap from telemetry — this can take a bit longer..."):
                df = compute_gap_on_track(session, driver)
            mode_label = "Gap to Car Ahead (actual on-track)"

        if df.empty:
            st.warning("No data available.")
        else:
            fig = go.Figure()

            has_ahead_info = "DriverAhead" in df.columns
            hovertemplate = "Lap %{x}<br>Gap: %{y:.3f} s"
            if has_ahead_info:
                hovertemplate += "<br>Ahead: %{customdata}"
            hovertemplate += "<extra></extra>"

            fig.add_trace(go.Scatter(
                x=df["Lap"],
                y=df["Gap"],
                mode="lines+markers",
                line=dict(color="#E10600", width=3), 
                marker=dict(size=5, color="#E10600"),
                fill="tozeroy",
                fillcolor="rgba(225, 6, 0, 0.15)",
                customdata=df["DriverAhead"] if has_ahead_info else None,
                hovertemplate=hovertemplate,
            ))

            fig.update_layout(
                title=dict(text=f"{driver} — {mode_label}", font=dict(size=22)),
                xaxis_title="Lap",
                yaxis_title="Gap (s)",
                template="plotly_white",
                hovermode="x unified",
                margin=dict(l=40, r=20, t=60, b=40),
                font=dict(family="Arial, sans-serif", size=13),
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

            st.plotly_chart(fig, use_container_width=True)
elif st.session_state.session_obj is not None and st.session_state.session_key != current_key:
    st.info("Year/Event/Session changed — click **Load Session** again to load the new selection.")
