"""Streamlit web app: Student Academic Performance Prediction (DNN).
Run:  streamlit run app.py   (after running  python train.py  once)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_DIR, OUT_DIR, PLOT_DIR = ROOT / "model", ROOT / "outputs", ROOT / "outputs" / "plots"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#0ea5e9 100%);
       padding: 2rem 2.2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.02rem;}
.result {border-radius: 18px; padding: 1.6rem; text-align: center; color: white;}
.result .label {font-size: .9rem; letter-spacing: .12em; text-transform: uppercase; opacity: .9;}
.result .value {font-size: 2.6rem; font-weight: 800; margin: .2rem 0;}
.result .desc {font-size: .98rem; opacity: .95;}
.bar-wrap {margin: .55rem 0;}
.bar-top {display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: 3px;}
.bar-bg {background: rgba(128,128,128,.2); border-radius: 8px; height: 12px; overflow: hidden;}
.bar-fg {height: 12px; border-radius: 8px;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 Student Academic Performance Predictor</h1>
  <p>A Deep Neural Network that predicts a student's final performance category
  from demographic, academic, social and behavioural information.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load artefacts
# ----------------------------------------------------------------------------
if not (MODEL_DIR / "dnn_model.keras").exists():
    st.error("Model not found. Run `python train.py` first to train and save the model.")
    st.stop()


@st.cache_resource
def load_artefacts():
    from tensorflow import keras
    model = keras.models.load_model(MODEL_DIR / "dnn_model.keras")
    pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
    info = json.loads((MODEL_DIR / "feature_info.json").read_text())
    return model, pre, info


model, pre, info = load_artefacts()
CLASSES, FEATURES = info["classes"], info["features"]

STYLE = {  # colour + meaning for each predicted category
    "Low":     ("linear-gradient(135deg,#ef4444,#f97316)", "#ef4444",
                "Final grade likely below 10. Needs academic support and intervention."),
    "Average": ("linear-gradient(135deg,#f59e0b,#eab308)", "#f59e0b",
                "Final grade likely between 10 and 14. Solid, with room to improve."),
    "High":    ("linear-gradient(135deg,#10b981,#059669)", "#10b981",
                "Final grade likely 15 or above. Strong academic and employability potential."),
}

LABELS = {  # label shown in the form (matches the project UI)
    "address": "Address (U=Urban, R=Rural)", "Medu": "Mother's education (0-4)",
    "Mjob": "Mother's job", "Fjob": "Father's job", "guardian": "Guardian",
    "G1": "First period grade (0-20)", "G2": "Second period grade (0-20)",
    "failures": "Past class failures", "schoolsup": "Extra school support",
    "higher": "Wants higher education",
    "famrel": "Family relationship quality (1-5)", "freetime": "Free time after school (1-5)",
    "goout": "Going out with friends (1-5)", "activities": "Extra-curricular activities",
    "internet": "Internet access at home",
    "studytime": "Weekly study time (1-4)", "absences": "Number of absences",
    "Dalc": "Weekday alcohol consumption (1-5)", "Walc": "Weekend alcohol consumption (1-5)",
    "health": "Health status (1-5)",
}
GROUP_ICONS = {"Demographic": "👤", "Academic": "📚", "Social": "👥", "Behavioural": "🧠"}
VALUE_NAMES = {"yes": "Yes", "no": "No", "U": "U (Urban)", "R": "R (Rural)", "at_home": "At home"}


def tab_label(f):
    return LABELS.get(f, f)


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_predict, tab_perf, tab_eda, tab_about = st.tabs(
    ["🎯 Predict", "📊 Model performance", "🔎 Data insights", "ℹ️ About"])

with tab_predict:
    st.subheader("Enter student details")
    with st.form("student_form"):
        values = {}
        cols = st.columns(4)
        for col, (group, feats) in zip(cols, info["groups"].items()):
            with col:
                st.markdown(f"#### {GROUP_ICONS.get(group, '')} {group}")
                for feat in feats:
                    m, label = FEATURES[feat], tab_label(feat)
                    if m["type"] == "categorical":
                        values[feat] = st.selectbox(
                            label, m["options"], index=m["options"].index(m["default"]),
                            format_func=lambda v: VALUE_NAMES.get(v, str(v).replace("_", " ").capitalize()))
                    else:
                        values[feat] = st.number_input(label, min_value=m["min"], max_value=m["max"],
                                                       value=m["default"], step=1)
        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        row = pd.DataFrame([values])[list(FEATURES.keys())]
        probs = model.predict(pre.transform(row), verbose=0)[0]
        pred = CLASSES[int(probs.argmax())]
        gradient, _, desc = STYLE[pred]

        c1, c2 = st.columns([1, 1.2], gap="large")
        with c1:
            st.markdown(f"""<div class="result" style="background:{gradient}">
                <div class="label">Predicted performance</div>
                <div class="value">{pred}</div>
                <div class="desc">{desc}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("**Model confidence**")
            for cls, p in zip(CLASSES, probs):
                st.markdown(f"""<div class="bar-wrap"><div class="bar-top"><span>{cls}</span>
                    <span><b>{p*100:.1f}%</b></span></div><div class="bar-bg">
                    <div class="bar-fg" style="width:{p*100:.1f}%;background:{STYLE[cls][1]}"></div>
                    </div></div>""", unsafe_allow_html=True)
    else:
        st.info("Fill in the student's details above and press **Predict**.")

with tab_perf:
    st.subheader("Deep Neural Network vs traditional Machine Learning")
    cmp_path = OUT_DIR / "model_comparison.csv"
    if cmp_path.exists():
        table = pd.read_csv(cmp_path)
        dnn = table[table["Model"] == "Deep Neural Network"].iloc[0]
        m = st.columns(4)
        for col, name in zip(m, ["Accuracy", "Precision", "Recall", "F1-score"]):
            col.metric(name, f"{dnn[name]*100:.1f}%")
        st.dataframe(table, hide_index=True)
    for img, cap in [("08_model_comparison.png", "Model comparison (test set)"),
                     ("07_confusion_matrices.png", "Confusion matrices"),
                     ("06_training_curves.png", "DNN training curves")]:
        if (PLOT_DIR / img).exists():
            st.image(str(PLOT_DIR / img), caption=cap)
    st.caption(f"Best DNN hyperparameters (random search): {info['best_params']}")

with tab_eda:
    st.subheader("Exploratory data analysis")
    for img, cap in [("01_target_distribution.png", "Target distribution"),
                     ("02_correlation_heatmap.png", "Correlation heatmap"),
                     ("03_key_relationships.png", "Key relationships with final grade"),
                     ("04_categorical_vs_target.png", "Categorical features vs performance"),
                     ("05_feature_importance.png", "Feature selection (mutual information)")]:
        if (PLOT_DIR / img).exists():
            st.image(str(PLOT_DIR / img), caption=cap)

with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
**Dataset:** UCI Student Performance dataset ({'Portuguese' if info['subject']=='por' else 'Mathematics'} course).
**Target:** final grade G3 converted into three categories: **Low** (<10), **Average** (10-14), **High** (15-20).

**Pipeline:** preprocessing → EDA → feature selection ({len(FEATURES)} features, ranked by mutual information) →
DNN design → random-search hyperparameter tuning → training with early stopping → evaluation →
comparison with Random Forest and Logistic Regression.

**Features used by the model:** {', '.join(tab_label(f) for f in FEATURES)}.
""")
