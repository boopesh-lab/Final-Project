"""
AI-Based Digital Twin Preventive Maintenance Monitoring System
================================================================
INTERACTIVE VERSION

You directly control sensor values with sliders. Every change you make
instantly recalculates the AI health score, risk level, Remaining Useful
Life (RUL), anomaly detection, and all graphs — so you can see exactly
how the system reacts as conditions change.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Twin | Preventive Maintenance",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# MACHINE DEFINITIONS (baseline / "healthy" operating values)
# --------------------------------------------------------------------------
MACHINES = {
    "PUMP-01":  {"type": "Centrifugal Pump", "temp": 65, "vib": 2.2, "press": 4.5, "rpm": 1450, "rpm_range": (1000, 2000)},
    "MOTOR-02": {"type": "Induction Motor",  "temp": 70, "vib": 1.8, "press": 0.0, "rpm": 1800, "rpm_range": (1200, 2400)},
    "COMP-03":  {"type": "Air Compressor",   "temp": 78, "vib": 3.0, "press": 7.2, "rpm": 3000, "rpm_range": (2000, 4000)},
    "TURB-04":  {"type": "Gas Turbine",      "temp": 95, "vib": 2.6, "press": 12.0, "rpm": 5200, "rpm_range": (4000, 6500)},
    "CONV-05":  {"type": "Conveyor Drive",   "temp": 55, "vib": 1.4, "press": 0.0, "rpm": 900,  "rpm_range": (500, 1400)},
}

SENSOR_LIMITS = {
    "temperature": {"warn": 85, "crit": 100, "max": 150},
    "vibration":   {"warn": 4.0, "crit": 6.0, "max": 10.0},
    "pressure":    {"warn": 9.0, "crit": 12.5, "max": 16.0},
}

RISK_COLORS = {"Low": "#2ecc71", "Moderate": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}
STATUS_COLORS = {"Normal": "#2ecc71", "Warning": "#f39c12", "Critical": "#e74c3c"}

HISTORY_LEN = 60  # points shown on trend graphs


# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
def init_session_state():
    if "slider_values" not in st.session_state:
        st.session_state.slider_values = {
            mid: {"temperature": cfg["temp"], "vibration": cfg["vib"],
                  "pressure": cfg["press"], "rpm": cfg["rpm"]}
            for mid, cfg in MACHINES.items()
        }
    if "history" not in st.session_state:
        st.session_state.history = {mid: pd.DataFrame(columns=["timestamp", "temperature", "vibration", "pressure", "rpm", "health_score"])
                                     for mid in MACHINES}
    if "maintenance_log" not in st.session_state:
        st.session_state.maintenance_log = []


# --------------------------------------------------------------------------
# AI / SCORING LOGIC — all driven purely by current slider values
# --------------------------------------------------------------------------
def compute_health(machine_id: str, vals: dict) -> dict:
    """Turns current sensor readings into a health score, risk level and
    RUL estimate. Pure function of the slider values -> changes instantly
    whenever you move a slider."""
    cfg = MACHINES[machine_id]

    # Normalized stress contribution from each sensor (0 = healthy, 1 = failing)
    temp_stress = np.clip((vals["temperature"] - cfg["temp"]) / (SENSOR_LIMITS["temperature"]["max"] - cfg["temp"]), 0, 1)
    vib_stress = np.clip((vals["vibration"] - cfg["vib"]) / (SENSOR_LIMITS["vibration"]["max"] - cfg["vib"]), 0, 1)
    press_stress = 0.0
    if cfg["press"] > 0:
        press_stress = np.clip((vals["pressure"] - cfg["press"]) / (SENSOR_LIMITS["pressure"]["max"] - cfg["press"]), 0, 1)
    rpm_dev = abs(vals["rpm"] - cfg["rpm"]) / cfg["rpm"]
    rpm_stress = np.clip(rpm_dev / 0.3, 0, 1)

    # Weighted composite degradation score
    degradation = (0.35 * temp_stress + 0.35 * vib_stress + 0.15 * press_stress + 0.15 * rpm_stress)
    degradation = float(np.clip(degradation, 0, 1))

    health_score = round((1 - degradation) * 100, 1)

    if degradation >= 0.75:
        risk = "Critical"
    elif degradation >= 0.5:
        risk = "High"
    elif degradation >= 0.25:
        risk = "Moderate"
    else:
        risk = "Low"

    # RUL heuristic: healthier machines -> longer projected life horizon
    max_horizon_hours = 2000
    rul_hours = round(max_horizon_hours * (1 - degradation) ** 2, 1)

    return {
        "degradation": degradation,
        "health_score": health_score,
        "risk": risk,
        "rul_hours": rul_hours,
        "temp_stress": temp_stress,
        "vib_stress": vib_stress,
        "press_stress": press_stress,
        "rpm_stress": rpm_stress,
    }


def sensor_status(vals: dict) -> dict:
    status = {}
    t = vals["temperature"]
    status["temperature"] = ("Critical" if t >= SENSOR_LIMITS["temperature"]["crit"]
                              else "Warning" if t >= SENSOR_LIMITS["temperature"]["warn"] else "Normal")
    v = vals["vibration"]
    status["vibration"] = ("Critical" if v >= SENSOR_LIMITS["vibration"]["crit"]
                            else "Warning" if v >= SENSOR_LIMITS["vibration"]["warn"] else "Normal")
    p = vals["pressure"]
    status["pressure"] = ("Critical" if p >= SENSOR_LIMITS["pressure"]["crit"]
                           else "Warning" if p >= SENSOR_LIMITS["pressure"]["warn"] else "Normal")
    return status


def detect_anomaly_live(machine_id: str, vals: dict) -> dict:
    """Fit an IsolationForest on the machine's recent recorded history
    (plus the current live reading) to flag whether the CURRENT slider
    reading looks anomalous relative to recent behavior."""
    hist = st.session_state.history[machine_id]
    feature_cols = ["temperature", "vibration", "pressure", "rpm"]
    current = pd.DataFrame([{**vals}])

    if len(hist) < 15:
        return {"is_anomaly": False, "score": 0.0, "confidence": "warming up (need more history)"}

    X_train = hist[feature_cols].values
    model = IsolationForest(n_estimators=150, contamination=0.1, random_state=42)
    model.fit(X_train)
    pred = model.predict(current[feature_cols].values)[0]
    raw_score = -model.score_samples(current[feature_cols].values)[0]

    train_scores = -model.score_samples(X_train)
    norm_score = float(np.clip((raw_score - train_scores.min()) / (train_scores.max() - train_scores.min() + 1e-9), 0, 1))

    return {"is_anomaly": pred == -1, "score": round(norm_score, 2), "confidence": "live model"}


def record_reading(machine_id: str, vals: dict, health_score: float):
    """Append the current slider-driven reading to this machine's history
    so trend graphs update live and the anomaly model has data to learn from."""
    row = {"timestamp": datetime.now(), **vals, "health_score": health_score}
    hist = st.session_state.history[machine_id]
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    if len(hist) > HISTORY_LEN:
        hist = hist.iloc[-HISTORY_LEN:].reset_index(drop=True)
    st.session_state.history[machine_id] = hist


# --------------------------------------------------------------------------
# CHART HELPERS
# --------------------------------------------------------------------------
def gauge_chart(value: float, title: str, max_val: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"font": {"size": 26}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, max_val * 0.5], "color": "#eafaf1"},
                {"range": [max_val * 0.5, max_val * 0.75], "color": "#fef5e7"},
                {"range": [max_val * 0.75, max_val], "color": "#fdecea"},
            ],
        },
    ))
    fig.update_layout(height=190, margin=dict(l=15, r=15, t=40, b=10))
    return fig


def stress_bar_chart(health: dict) -> go.Figure:
    labels = ["Temperature", "Vibration", "Pressure", "RPM Deviation"]
    values = [health["temp_stress"] * 100, health["vib_stress"] * 100,
              health["press_stress"] * 100, health["rpm_stress"] * 100]
    colors = ["#e74c3c" if v >= 70 else "#f39c12" if v >= 40 else "#2ecc71" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                            text=[f"{v:.0f}%" for v in values], textposition="outside"))
    fig.update_layout(title="Contribution to Degradation Score", height=260,
                       xaxis=dict(range=[0, 110], title="Stress %"), margin=dict(l=10, r=10, t=40, b=10))
    return fig


def trend_chart(hist: pd.DataFrame, col: str, label: str, unit: str, warn=None, crit=None) -> go.Figure:
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist[col], mode="lines+markers",
                                  name=label, line=dict(width=2, color="#2980b9"), marker=dict(size=4)))
    if warn is not None:
        fig.add_hline(y=warn, line_dash="dot", line_color="#f39c12", annotation_text="Warn")
    if crit is not None:
        fig.add_hline(y=crit, line_dash="dot", line_color="#e74c3c", annotation_text="Critical")
    fig.update_layout(title=f"{label} Trend ({unit})", height=250, margin=dict(l=10, r=10, t=40, b=10),
                       showlegend=False, xaxis_title=None, yaxis_title=unit)
    return fig


def health_trend_chart(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["health_score"], mode="lines+markers",
                                  fill="tozeroy", line=dict(width=2, color="#8e44ad"), marker=dict(size=4)))
    fig.update_layout(title="Health Score Trend (%)", height=250, margin=dict(l=10, r=10, t=40, b=10),
                       yaxis=dict(range=[0, 100]), xaxis_title=None)
    return fig


# --------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------
def main():
    init_session_state()

    # ---------------- Sidebar: machine picker + live sliders ----------------
    st.sidebar.title("🛠️ Digital Twin Control")
    st.sidebar.caption("Move the sliders below to simulate live sensor readings. "
                        "Every change instantly updates the health score, risk, RUL, "
                        "anomaly detection, and all graphs.")

    selected_machine = st.sidebar.selectbox("Select Machine", list(MACHINES.keys()),
                                             format_func=lambda m: f"{m} — {MACHINES[m]['type']}")
    cfg = MACHINES[selected_machine]
    sv = st.session_state.slider_values[selected_machine]

    st.sidebar.markdown("### 🎚️ Sensor Inputs")
    sv["temperature"] = st.sidebar.slider("Temperature (°C)", 0.0, float(SENSOR_LIMITS["temperature"]["max"]),
                                           float(sv["temperature"]), 0.5)
    sv["vibration"] = st.sidebar.slider("Vibration (mm/s)", 0.0, float(SENSOR_LIMITS["vibration"]["max"]),
                                         float(sv["vibration"]), 0.1)
    if cfg["press"] > 0:
        sv["pressure"] = st.sidebar.slider("Pressure (bar)", 0.0, float(SENSOR_LIMITS["pressure"]["max"]),
                                            float(sv["pressure"]), 0.1)
    else:
        sv["pressure"] = 0.0
        st.sidebar.caption("Pressure: not applicable for this machine")
    rmin, rmax = cfg["rpm_range"]
    sv["rpm"] = st.sidebar.slider("RPM", float(rmin) * 0.5, float(rmax) * 1.3, float(sv["rpm"]), 10.0)

    colA, colB = st.sidebar.columns(2)
    with colA:
        if st.button("↩️ Reset to healthy", use_container_width=True):
            sv["temperature"], sv["vibration"], sv["pressure"], sv["rpm"] = cfg["temp"], cfg["vib"], cfg["press"], cfg["rpm"]
    with colB:
        if st.button("🎲 Randomize fault", use_container_width=True):
            rng = np.random.default_rng()
            sv["temperature"] = cfg["temp"] + rng.uniform(5, 40)
            sv["vibration"] = cfg["vib"] + rng.uniform(0.5, 5)
            sv["pressure"] = cfg["press"] + (rng.uniform(0.5, 4) if cfg["press"] > 0 else 0)
            sv["rpm"] = cfg["rpm"] * rng.uniform(0.75, 1.3)

    record_now = st.sidebar.toggle("📌 Record every change to history/graphs", value=True,
                                    help="When on, each slider adjustment is appended as a new data point on the trend graphs.")

    if st.sidebar.button("🧾 Log maintenance performed"):
        sv["temperature"], sv["vibration"], sv["pressure"], sv["rpm"] = cfg["temp"], cfg["vib"], cfg["press"], cfg["rpm"]
        st.session_state.maintenance_log.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine": selected_machine,
            "action": "Preventive maintenance performed — sensors reset to healthy baseline",
        })
        st.sidebar.success(f"{selected_machine} serviced.")

    # ---------------- Compute everything from current slider state ----------------
    health = compute_health(selected_machine, sv)
    status = sensor_status(sv)
    anomaly = detect_anomaly_live(selected_machine, sv)

    if record_now:
        record_reading(selected_machine, sv, health["health_score"])

    hist = st.session_state.history[selected_machine]

    # ---------------- Header ----------------
    st.title("🛠️ AI-Based Digital Twin — Preventive Maintenance Dashboard")
    st.caption("Interactive simulation — adjust sliders in the sidebar and watch the outputs and graphs change live.")

    st.subheader(f"🔍 {selected_machine} — {cfg['type']}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Health Score", f"{health['health_score']}%",
              delta=f"{health['health_score'] - 100:.1f}% vs baseline")
    k2.metric("Risk Level", health["risk"])
    k3.metric("Est. RUL", f"{health['rul_hours']} h")
    k4.metric("AI Anomaly Score", f"{anomaly['score']*100:.0f}%",
              delta="⚠ ANOMALY" if anomaly["is_anomaly"] else "normal", delta_color="inverse")

    # ---------------- Gauges ----------------
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(gauge_chart(health["health_score"], "Health Score (%)", 100, RISK_COLORS[health["risk"]]),
                         use_container_width=True)
    with g2:
        st.plotly_chart(gauge_chart(sv["temperature"], "Temperature (°C)", SENSOR_LIMITS["temperature"]["max"],
                                     STATUS_COLORS[status["temperature"]]), use_container_width=True)
    with g3:
        st.plotly_chart(gauge_chart(sv["vibration"], "Vibration (mm/s)", SENSOR_LIMITS["vibration"]["max"],
                                     STATUS_COLORS[status["vibration"]]), use_container_width=True)
    with g4:
        if cfg["press"] > 0:
            st.plotly_chart(gauge_chart(sv["pressure"], "Pressure (bar)", SENSOR_LIMITS["pressure"]["max"],
                                         STATUS_COLORS[status["pressure"]]), use_container_width=True)
        else:
            st.plotly_chart(gauge_chart(sv["rpm"], "RPM", cfg["rpm_range"][1] * 1.3, "#3498db"), use_container_width=True)

    # ---------------- Stress breakdown ----------------
    st.plotly_chart(stress_bar_chart(health), use_container_width=True)

    # ---------------- Alerts ----------------
    st.markdown("#### 🚨 Live Alerts")
    alerts = []
    for sensor, s in status.items():
        if s in ("Warning", "Critical") and (sensor != "pressure" or cfg["press"] > 0):
            alerts.append((s, f"{sensor.capitalize()} is {s.upper()} ({sv[sensor]:.1f})"))
    if health["risk"] in ("High", "Critical"):
        alerts.append((health["risk"], f"Composite risk is {health['risk']} — estimated RUL only {health['rul_hours']} hours."))
    if anomaly["is_anomaly"]:
        alerts.append(("Warning", f"AI model flags current reading as anomalous relative to recent history (score {anomaly['score']*100:.0f}%)."))

    if not alerts:
        st.success("✅ No active alerts. Machine operating within normal parameters.")
    else:
        for level, msg in alerts:
            if level == "Critical":
                st.error(f"🔴 {msg}")
            else:
                st.warning(f"🟠 {msg}")

    # ---------------- Trend graphs (update as history grows) ----------------
    st.markdown("#### 📈 Live Trend Graphs")
    if hist.empty:
        st.info("Move a slider (with recording on) to start building the trend graphs.")
    else:
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.plotly_chart(trend_chart(hist, "temperature", "Temperature", "°C",
                                         SENSOR_LIMITS["temperature"]["warn"], SENSOR_LIMITS["temperature"]["crit"]),
                             use_container_width=True)
            st.plotly_chart(health_trend_chart(hist), use_container_width=True)
        with tcol2:
            st.plotly_chart(trend_chart(hist, "vibration", "Vibration", "mm/s",
                                         SENSOR_LIMITS["vibration"]["warn"], SENSOR_LIMITS["vibration"]["crit"]),
                             use_container_width=True)
            st.plotly_chart(trend_chart(hist, "rpm", "RPM", "rpm"), use_container_width=True)

    # ---------------- Fleet comparison snapshot ----------------
    with st.expander("🏭 Compare all machines at their current slider settings"):
        rows = []
        for mid, mcfg in MACHINES.items():
            mvals = st.session_state.slider_values[mid]
            mh = compute_health(mid, mvals)
            rows.append({"Machine": mid, "Type": mcfg["type"], "Health %": mh["health_score"],
                         "Risk": mh["risk"], "RUL (h)": mh["rul_hours"]})
        fleet_df = pd.DataFrame(rows)
        st.dataframe(fleet_df, use_container_width=True, hide_index=True)
        fig = px.bar(fleet_df, x="Machine", y="Health %", color="Risk", color_discrete_map=RISK_COLORS)
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Maintenance log ----------------
    with st.expander("🧾 Maintenance log"):
        if st.session_state.maintenance_log:
            st.dataframe(pd.DataFrame(st.session_state.maintenance_log), use_container_width=True, hide_index=True)
        else:
            st.info("No maintenance actions logged yet.")

    # ---------------- Raw history table ----------------
    with st.expander("🗂️ Recorded history (this machine)"):
        if hist.empty:
            st.info("No history recorded yet.")
        else:
            st.dataframe(hist.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
