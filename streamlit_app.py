import streamlit as st
import pandas as pd
import json
import joblib

st.set_page_config(page_title="Beat Forecast", layout="wide")

# ---------------- Clustering Helpers ----------------
@st.cache_resource
def load_clustering_assets():
    scaler = joblib.load("cluster_scaler.pkl")
    kmeans = joblib.load("kmeans_k6.pkl")
    with open("cluster_feature_list.json", "r") as f:
        feats = json.load(f)

    labels = pd.read_csv("cluster_labels.csv")  # cluster,name,description
    hit_summary = pd.read_csv("cluster_hit_summary.csv")  # cluster,name,hit_rate,...
    try:
        top2024 = pd.read_csv("cluster_top2024_lift.csv")  # name, lift, etc
    except Exception:
        top2024 = None

    return scaler, kmeans, feats, labels, hit_summary, top2024


# ---------------- GB Hit Model (Classification) ----------------
@st.cache_resource
def load_gb_hit_model():
    return joblib.load("models/hit_gb_pipeline_wade.joblib")


def predict_archetype(user_inputs: dict):
    scaler, kmeans, feats, labels, hit_summary, top2024 = load_clustering_assets()

    X = pd.DataFrame([{f: user_inputs.get(f) for f in feats}]).reindex(columns=feats)
    Xs = scaler.transform(X)
    cid = int(kmeans.predict(Xs)[0])

    lab = labels[labels["cluster"] == cid].head(1)
    name = lab["name"].iloc[0] if len(lab) else f"Cluster {cid}"
    desc = lab["description"].iloc[0] if len(lab) else ""

    hs = hit_summary[hit_summary["cluster"] == cid].head(1)
    t24 = top2024[top2024["name"] == name].head(1) if top2024 is not None else None

    return cid, name, desc, hs, t24


# ---------------- Header ----------------
st.title("Beat Forecast")
st.caption("Data-driven decision support for estimating Spotify performance prior to release.")
st.divider()

# ---------------- Market Context ----------------
col_mc1, col_mc2 = st.columns([2, 1])

with col_mc1:
    st.subheader("Market Context")
    st.write(
        "Spotify breakout success is structurally rare. "
        "Only a small fraction of tracks achieve top-tier performance. "
        "This tool evaluates whether a song aligns with structural success patterns."
    )

with col_mc2:
    st.metric("Observed Hit Rate (2024)", "0.22%")

st.divider()

# ---------------- Sidebar ----------------
st.sidebar.header("Model Inputs")

with st.sidebar.expander("Artist Context", expanded=True):
    followers = st.number_input("Spotify Followers", min_value=0, value=5000, step=100)
    artist_pop = st.slider("Artist Popularity (0–100)", 0, 100, 35)

with st.sidebar.expander("Audio Features", expanded=True):
    danceability = st.slider("Danceability", 0.0, 1.0, 0.60)
    energy = st.slider("Energy", 0.0, 1.0, 0.70)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -7.0)
    tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.50)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.10)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.10)
    key = st.selectbox("Key (0–11)", list(range(12)), index=5)

    mode_label = st.selectbox("Mode", ["Minor (0)", "Major (1)"], index=1)
    mode = 0 if mode_label.startswith("Minor") else 1
    # required for clustering model
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
    duration_ms = st.number_input("Duration (ms)", min_value=0, value=210000, step=1000)

st.sidebar.divider()

# Persist run state so results stay visible after click
if "run_forecast" not in st.session_state:
    st.session_state["run_forecast"] = False

def do_run():
    st.session_state["run_forecast"] = True

def do_reset():
    st.session_state["run_forecast"] = False

st.sidebar.button("Run Forecast", type="primary", on_click=do_run)
st.sidebar.button("Reset", on_click=do_reset)

run = st.session_state["run_forecast"]

# ---------------- Main Layout ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Breakout Probability")
    hit_metric = st.empty()

with col2:
    st.subheader("Popularity Forecast")
    pop_metric = st.empty()

with col3:
    st.subheader("Executive Recommendation")
    rec_box = st.empty()
    rec_box.info("Model connection pending. Recommendation will appear here.")

st.divider()

# ---------------- Input Summary ----------------
st.subheader("Input Summary")

inputs = pd.DataFrame([{
    "danceability": danceability,
    "energy": energy,
    "loudness": loudness,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "duration_ms": duration_ms,
    "key": key,
    "mode": mode,
    "total_artist_followers": followers,
}])

st.dataframe(inputs, use_container_width=True, hide_index=True)

st.divider()

# ---------------- Scenario Comparison ----------------
st.subheader("Scenario Comparison")

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = []

colS1, colS2 = st.columns([1, 1])

with colS1:
    if st.button("Save Current Scenario"):
        st.session_state["scenarios"].append(inputs.iloc[0].to_dict())

with colS2:
    if st.button("Clear Scenarios"):
        st.session_state["scenarios"] = []

if st.session_state["scenarios"]:
    st.dataframe(pd.DataFrame(st.session_state["scenarios"]), use_container_width=True, hide_index=True)
else:
    st.caption("Save multiple input sets to compare tradeoffs across songs or mixes.")

st.divider()

# ---------------- Drivers ----------------
st.subheader("Performance Drivers")
st.write("Directional guidance will appear here. Model-based feature importance will be added during final integration.")

# ---------------- Run Forecast ----------------
if run:
    st.success("RUN BLOCK FIRED ✅")  # remove later if you want

    # ----- GB model scoring (classification) -----
    gb_model = load_gb_hit_model()

    # IMPORTANT: column names must match training
    # If your model expects artist_pop instead of artist_popularity, we rename here:
    X_input = inputs.copy()
   
    feature_cols = gb_model.feature_names_in_.tolist()
X_input = inputs.reindex(columns=feature_cols)
breakout_prob = float(gb_model.predict_proba(X_input)[:, 1][0])

breakout_prob = float(gb_model.predict_proba(X_input)[:, 1][0])

hit_metric.metric("Hit Likelihood", f"{round(breakout_prob * 100, 1)}%")

    # Keep popularity forecast as placeholder until regression gets wired
predicted_popularity = 20 + (breakout_prob * 80)
pop_metric.metric("Predicted Popularity (0–100)", round(predicted_popularity, 1))

if breakout_prob > 0.40:
        recommendation = "Release-ready under current production and exposure profile."
        rec_box.success(recommendation)
elif breakout_prob > 0.20:
        recommendation = "Moderate breakout potential. Consider production refinements or stronger promotion."
        rec_box.warning(recommendation)
else:
        recommendation = "Low projected breakout probability under current inputs."
        rec_box.error(recommendation)

st.divider()

    # ----- ✅ Clustering Archetype -----
st.subheader("Song Archetype (Clustering)")

cluster_inputs = {
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "tempo": tempo,
        "loudness": loudness,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "duration_ms": duration_ms,
    }

try:
        cid, archetype, archetype_desc, hs, t24 = predict_archetype(cluster_inputs)
        st.write(f"**{archetype}**")
        if archetype_desc:
            st.caption(archetype_desc)

        if len(hs):
            c1, c2, c3 = st.columns(3)
            c1.metric("Hit rate (pop≥70)", f"{hs['hit_rate'].iloc[0]:.3f}")

            rel_col = (
                "relative_to_overall"
                if "relative_to_overall" in hs.columns
                else ("enrichment_ratio" if "enrichment_ratio" in hs.columns else None)
            )
            if rel_col:
                c2.metric("Relative to overall", f"{hs[rel_col].iloc[0]:.2f}×")
            else:
                c2.metric("Relative to overall", "—")

            c3.metric("Avg popularity", f"{hs['avg_popularity'].iloc[0]:.1f}" if "avg_popularity" in hs.columns else "—")

        if t24 is not None and len(t24) and "lift" in t24.columns:
            st.metric("Top-2024 lift", f"{t24['lift'].iloc[0]:.2f}×")

except Exception as e:
        st.error(f"Clustering could not run. Check that all clustering files exist and match expected names. Details: {e}")

st.divider()

    # ----- Production Assessment (kept) -----
st.subheader("Production Assessment")

strengths, weaknesses = [], []

if energy > 0.65:
        strengths.append("Energy aligns with high-performing tracks.")
else:
        weaknesses.append("Energy below common hit threshold (0.65).")

if danceability > 0.60:
        strengths.append("Danceability within competitive streaming range.")
else:
        weaknesses.append("Danceability below common engagement range (0.60).")

if loudness > -8:
        strengths.append("Loudness consistent with commercial production standards.")
else:
        weaknesses.append("Loudness below competitive streaming levels (-8 dB threshold).")

if followers > 50000:
        strengths.append("Strong baseline artist reach.")
else:
        weaknesses.append("Limited artist reach may constrain exposure.")

colA, colB = st.columns(2)

with colA:
        st.markdown("**Strengths**")
        if strengths:
            for s in strengths:
                st.write("-", s)
        else:
            st.write("No structural strengths identified.")

with colB:
        st.markdown("**Areas for Improvement**")
        if weaknesses:
            for w in weaknesses:
                st.write("-", w)
        else:
            st.write("No material weaknesses identified.")

st.divider()
st.subheader("Signal Strength")
st.write("Breakout probability visualization based on structural alignment.")
st.progress(breakout_prob)