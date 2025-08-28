# dashboard_sim.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json, joblib, random

st.set_page_config(page_title="Network Traffic — 1-Row Simulator", layout="wide")
st.title("🚀 Network Traffic ML — Single-Row Simulator (Gradient Boosting)")

MODELS_DIR = Path("models")
META_PATH = MODELS_DIR / "training_metadata.json"

# ---- Load metadata + model ----
if not META_PATH.exists():
    st.error("models/training_metadata.json not found. Train models first.")
    st.stop()

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

gb_path = MODELS_DIR / meta["models"].get("gradient_boosting", "gradient_boosting_model.pkl")
if not gb_path.exists():
    gb_path = MODELS_DIR / "gradient_boosting_model.pkl"
if not gb_path.exists():
    st.error("Gradient Boosting model .pkl not found in ./models. Train models first.")
    st.stop()

model = joblib.load(gb_path)
feature_cols = meta["feature_columns"]

st.caption(f"Model: **{gb_path.name}** • Features: **{len(feature_cols)}**")

# ---- Helpers ----
def align_single_row(d: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Create a 1-row DataFrame with the exact training feature schema/order."""
    row = {c: 0.0 for c in feature_cols}              # default zeros for missing
    for k, v in d.items():
        if k in row:
            try:
                row[k] = float(v)
            except Exception:
                row[k] = 0.0
    X = pd.DataFrame([row], columns=feature_cols).astype(float)
    return X

def preset_baseline(feature_cols: list[str]) -> dict:
    """Reasonable low/no-traffic baseline."""
    d = {c: 0.0 for c in feature_cols}
    # common numeric hints if present
    for key in [
        "total_packets","avg_packet_size","total_payload","tcp_ratio","udp_ratio",
        "icmp_ratio","port_scan_intensity","ttl_variance","temporal_duration",
        "top_dst_port","top_dst_port_share"
    ]:
        if key in d: d[key] = 0.0
    # one-hot service defaults (e.g., port0)
    for c in feature_cols:
        if c.startswith("svc_"):
            d[c] = 0.0
    if "svc_port0" in d:
        d["svc_port0"] = 1.0
    return d

def preset_small_dns_burst(feature_cols: list[str]) -> dict:
    d = preset_baseline(feature_cols)
    if "total_packets" in d: d["total_packets"] = 150
    if "udp_ratio" in d: d["udp_ratio"] = 0.98
    if "tcp_ratio" in d: d["tcp_ratio"] = 0.02
    if "avg_packet_size" in d: d["avg_packet_size"] = 80
    if "top_dst_port" in d: d["top_dst_port"] = 53
    if "top_dst_port_share" in d: d["top_dst_port_share"] = 0.9
    # service one-hots
    for c in feature_cols:
        if c.startswith("svc_"): d[c] = 0.0
    for key in ["svc_dns"]:
        if key in d: d[key] = 1.0
    return d

def randomize_small_subset(d: dict, keys: list[str], scale: float = 1.0) -> dict:
    """Jitter a small subset of numeric fields."""
    for k in keys:
        if k in d:
            base = float(d[k])
            d[k] = max(0.0, base + random.uniform(-1, 1) * scale * (1 + abs(base)))
    return d

# ---- UI: presets and editing ----
st.subheader("1) Build a single synthetic row")

colL, colR = st.columns([1, 1])

with colL:
    st.markdown("**Presets**")
    if st.button("🧪 Baseline (port0 / zeros)"):
        st.session_state.row = preset_baseline(feature_cols)
    if st.button("🧪 Small DNS burst"):
        st.session_state.row = preset_small_dns_burst(feature_cols)
    if st.button("🎲 Randomize a few metrics"):
        if "row" not in st.session_state:
            st.session_state.row = preset_baseline(feature_cols)
        keys_to_jitter = [k for k in ["total_packets","avg_packet_size","total_payload",
                                      "tcp_ratio","udp_ratio","icmp_ratio",
                                      "port_scan_intensity","ttl_variance",
                                      "top_dst_port_share"] if k in feature_cols]
        st.session_state.row = randomize_small_subset(st.session_state.row, keys_to_jitter, scale=0.5)

if "row" not in st.session_state:
    st.session_state.row = preset_baseline(feature_cols)

with colR:
    st.markdown("**Quick edit (first 12 features)**")
    preview_keys = feature_cols[:12]
    edits = {}
    grid = st.columns(3)
    for i, k in enumerate(preview_keys):
        v = st.session_state.row.get(k, 0.0)
        edits[k] = grid[i % 3].number_input(k, value=float(v))
    # apply edits
    st.session_state.row.update(edits)

# ---- Predict ----
st.subheader("2) Predict")
X = align_single_row(st.session_state.row, feature_cols)
st.code(X.to_string(index=False), language="text")

if st.button("🔮 Predict this row"):
    pred = model.predict(X)[0]
    st.success(f"Prediction: **{pred}**")
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                st.write("Probabilities:")
                st.json({str(c): float(p) for c, p in zip(classes, probs)})
        except Exception as e:
            st.warning(f"Could not compute probabilities: {e}")
