"""
Student Academic Performance Prediction using a Deep Neural Network (DNN)
=========================================================================
Run:  python train.py

Pipeline (matches the project brief):
  1. Data loading & preprocessing
  2. Exploratory data analysis (plots saved to outputs/plots)
  3. Feature selection (20 chosen features; mutual-information ranking on training data)
  4. DNN architecture design
  5. Hyperparameter tuning (random search)
  6. Model training (early stopping, class weights)
  7. Performance evaluation on an unseen test set
  8. Comparison with traditional ML (Random Forest, Logistic Regression)
  9. Saving artefacts used by the Streamlit app (app.py)
"""
import io
import json
import random
import urllib.request
import warnings
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
SUBJECT = "por"          # "por" = Portuguese (649 rows) | "mat" = Mathematics (395 rows)
# Features used by the model (and asked in the web app), grouped as in the UI
FEATURE_GROUPS = {
    "Demographic": ["address", "Medu", "Mjob", "Fjob", "guardian"],
    "Academic":    ["G1", "G2", "failures", "schoolsup", "higher"],
    "Social":      ["famrel", "freetime", "goout", "activities", "internet"],
    "Behavioural": ["studytime", "absences", "Dalc", "Walc", "health"],
}
SELECTED_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
N_TRIALS = 12            # random-search trials for hyperparameter tuning
MAX_EPOCHS = 200

# Target: final grade G3 (0-20) converted to a performance category
CLASSES = ["Low", "Average", "High"]          # G3 < 10 | 10-14 | 15-20


def categorical_cols(df):
    """Non-numeric columns (works for both 'object' and pandas>=3 'str' dtypes)."""
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def grade_to_category(g3):
    return 0 if g3 < 10 else (1 if g3 < 15 else 2)


ROOT = Path(__file__).parent
DATA_DIR, OUT_DIR = ROOT / "data", ROOT / "outputs"
PLOT_DIR, MODEL_DIR = OUT_DIR / "plots", ROOT / "model"
for d in (DATA_DIR, PLOT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)


# ----------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ----------------------------------------------------------------------------
UCI_ZIP = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def _extract_zip(raw: bytes):
    """Extract a zip (and any zips nested inside it) into data/."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            data = z.read(name)
            if name.lower().endswith(".zip"):
                _extract_zip(data)
            elif name.lower().endswith(".csv"):
                (DATA_DIR / Path(name).name).write_bytes(data)


def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / f"student-{SUBJECT}.csv"
    if not csv_path.exists():
        print(f"Downloading dataset from UCI ...")
        try:
            raw = urllib.request.urlopen(UCI_ZIP, timeout=60).read()
            _extract_zip(raw)
        except Exception as e:
            raise SystemExit(
                f"Could not download the dataset ({e}).\n"
                f"Download it manually from the UCI link in the brief and put "
                f"'student-{SUBJECT}.csv' inside the 'data/' folder."
            )
    df = pd.read_csv(csv_path, sep=";")
    if df.shape[1] == 1:                      # fallback if comma separated
        df = pd.read_csv(csv_path, sep=",")
    print(f"Loaded {csv_path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Data preprocessing ---")
    print("Missing values :", int(df.isna().sum().sum()))
    print("Duplicate rows :", int(df.duplicated().sum()))
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    df["performance"] = df["G3"].apply(grade_to_category)
    print("Class distribution:\n", df["performance"].map(dict(enumerate(CLASSES))).value_counts())
    return df


# ----------------------------------------------------------------------------
# 2. Exploratory data analysis
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    print("\n--- Exploratory data analysis ---")
    df.describe().T.to_csv(OUT_DIR / "dataset_summary.csv")
    colors = ["#ef4444", "#f59e0b", "#10b981"]

    # (a) G3 distribution + category counts
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["G3"], bins=range(0, 22), color="#6366f1", edgecolor="white")
    ax[0].set(title="Distribution of final grade (G3)", xlabel="G3", ylabel="Students")
    counts = df["performance"].value_counts().sort_index()
    ax[1].bar(CLASSES, counts.values, color=colors)
    ax[1].set(title="Performance categories", ylabel="Students")
    for i, v in enumerate(counts.values):
        ax[1].text(i, v + 2, str(v), ha="center")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "01_target_distribution.png", dpi=130); plt.close(fig)

    # (b) correlation heatmap of numeric columns
    num = df.select_dtypes("number").drop(columns=["performance"])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    fig.colorbar(im); ax.set_title("Correlation heatmap (numeric features)")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "02_correlation_heatmap.png", dpi=130); plt.close(fig)

    # (c) key relationships with the final grade
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    df.groupby("studytime")["G3"].mean().plot.bar(ax=ax[0, 0], color="#6366f1")
    ax[0, 0].set(title="Average G3 by weekly study time", xlabel="Study time (1=<2h ... 4=>10h)", ylabel="Mean G3")
    df.groupby("failures")["G3"].mean().plot.bar(ax=ax[0, 1], color="#ef4444")
    ax[0, 1].set(title="Average G3 by past failures", xlabel="Past failures", ylabel="Mean G3")
    ax[1, 0].scatter(df["absences"], df["G3"], alpha=.5, color="#0ea5e9")
    ax[1, 0].set(title="Absences vs G3", xlabel="Absences", ylabel="G3")
    ax[1, 1].scatter(df["G2"], df["G3"], alpha=.5, color="#10b981")
    ax[1, 1].set(title="Period-2 grade (G2) vs G3", xlabel="G2", ylabel="G3")
    for a in ax.ravel():
        a.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "03_key_relationships.png", dpi=130); plt.close(fig)

    # (d) categorical features vs performance category
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col in zip(ax, ["higher", "schoolsup", "internet"]):
        pd.crosstab(df[col], df["performance"].map(dict(enumerate(CLASSES)))) \
            [CLASSES].plot.bar(ax=a, color=colors, rot=0)
        a.set(title=f"'{col}' vs performance", xlabel="")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "04_categorical_vs_target.png", dpi=130); plt.close(fig)
    print("EDA plots saved to outputs/plots/")


# ----------------------------------------------------------------------------
# 3. Feature selection
# ----------------------------------------------------------------------------
def select_features(X_train: pd.DataFrame, y_train, chosen):
    """Rank ALL attributes by mutual information with the target (training data only)
    and keep the chosen feature set. The ranking plot shows how the chosen features score."""
    print("\n--- Feature selection ---")
    enc = X_train.copy()
    cats = categorical_cols(X_train)
    for c in cats:
        enc[c] = enc[c].astype("category").cat.codes
    discrete = np.array([c in cats or X_train[c].nunique() <= 10 for c in X_train.columns])
    mi = mutual_info_classif(enc, y_train, discrete_features=discrete, random_state=SEED)
    scores = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    scores.round(4).to_csv(OUT_DIR / "feature_mutual_information.csv", header=["mutual_information"])
    print(f"Using {len(chosen)} features:", chosen)

    fig, ax = plt.subplots(figsize=(8, 9))
    top = scores[::-1]
    ax.barh(top.index, top.values,
            color=["#6366f1" if f in chosen else "#cbd5e1" for f in top.index])
    ax.set(title="Mutual information with performance\n(purple = features used by the model)",
           xlabel="MI score")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "05_feature_importance.png", dpi=130); plt.close(fig)
    return list(chosen)


def build_preprocessor(X: pd.DataFrame):
    cat = categorical_cols(X)
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def feature_metadata(X: pd.DataFrame) -> dict:
    """Information the Streamlit app needs to build its input form."""
    meta = {}
    for c in X.columns:
        if c in categorical_cols(X):
            meta[c] = {"type": "categorical", "options": sorted(X[c].unique().tolist()),
                       "default": X[c].mode()[0]}
        else:
            meta[c] = {"type": "numeric", "min": int(X[c].min()), "max": int(X[c].max()),
                       "default": int(X[c].median())}
    return meta


# ----------------------------------------------------------------------------
# 4. DNN architecture + 5. hyperparameter tuning + 6. training
# ----------------------------------------------------------------------------
def build_dnn(input_dim, n_layers, units, dropout, lr):
    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    for i in range(n_layers):
        model.add(layers.Dense(max(units // (2 ** i), 8), activation="relu",
                               kernel_regularizer=keras.regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(len(CLASSES), activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_dnn(params, Xtr, ytr, Xval, yval, class_weight, verbose=0):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = build_dnn(Xtr.shape[1], params["n_layers"], params["units"],
                      params["dropout"], params["lr"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                         restore_best_weights=True)
    reduce = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8)
    hist = model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=MAX_EPOCHS,
                     batch_size=params["batch_size"], class_weight=class_weight,
                     callbacks=[stop, reduce], verbose=verbose)
    return model, hist


def tune_dnn(Xtr, ytr, Xval, yval, class_weight):
    print(f"\n--- Hyperparameter tuning: random search, {N_TRIALS} trials ---")
    space = {"n_layers": [2, 3, 4], "units": [32, 64, 128, 256],
             "dropout": [0.1, 0.2, 0.3, 0.4], "lr": [1e-2, 5e-3, 1e-3, 5e-4],
             "batch_size": [16, 32, 64]}
    rng = random.Random(SEED)
    results, best = [], None
    for t in range(1, N_TRIALS + 1):
        params = {k: rng.choice(v) for k, v in space.items()}
        model, _ = train_dnn(params, Xtr, ytr, Xval, yval, class_weight)
        pred = model.predict(Xval, verbose=0).argmax(1)
        f1 = f1_score(yval, pred, average="macro")
        acc = accuracy_score(yval, pred)
        print(f"Trial {t:2d}: {params} -> val acc {acc:.3f} | val macro-F1 {f1:.3f}")
        results.append({**params, "val_accuracy": acc, "val_macro_f1": f1})
        if best is None or (f1, acc) > (best[0], best[1]):
            best = (f1, acc, params)
    pd.DataFrame(results).sort_values("val_macro_f1", ascending=False) \
        .to_csv(OUT_DIR / "hyperparameter_search.csv", index=False)
    print("Best hyperparameters:", best[2])
    return best[2]


# ----------------------------------------------------------------------------
# 7. Evaluation + 8. comparison with traditional ML
# ----------------------------------------------------------------------------
def metrics_row(name, y_true, y_pred):
    return {"Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1-score": f1_score(y_true, y_pred, average="macro")}


def main():
    df = preprocess(load_data())
    run_eda(df)

    # Train / validation / test split (70 / 15 / 15, stratified)
    X = df.drop(columns=["G3", "performance"])
    y = df["performance"].values
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f"\nSplit -> train {len(X_tr)} | validation {len(X_val)} | test {len(X_te)}")

    selected = select_features(X_tr, y_tr, SELECTED_FEATURES)
    X_tr, X_val, X_te = X_tr[selected], X_val[selected], X_te[selected]

    pre = build_preprocessor(X_tr)
    Xtr, Xval, Xte = (pre.fit_transform(X_tr), pre.transform(X_val), pre.transform(X_te))
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)))

    # DNN: tune -> retrain with best params -> evaluate
    best = tune_dnn(Xtr, y_tr, Xval, y_val, cw)
    model, hist = train_dnn(best, Xtr, y_tr, Xval, y_val, cw, verbose=0)
    model.summary()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="validation")
    ax[0].set(title="DNN loss", xlabel="Epoch"); ax[0].legend()
    ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="validation")
    ax[1].set(title="DNN accuracy", xlabel="Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(PLOT_DIR / "06_training_curves.png", dpi=130); plt.close(fig)

    preds = {"Deep Neural Network": model.predict(Xte, verbose=0).argmax(1)}
    baselines = {
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }
    for name, clf in baselines.items():
        clf.fit(Xtr, y_tr)
        preds[name] = clf.predict(Xte)

    table = pd.DataFrame([metrics_row(n, y_te, p) for n, p in preds.items()]).round(4)
    print("\n--- Test-set comparison ---\n", table.to_string(index=False))
    table.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\nDNN classification report:\n",
          classification_report(y_te, preds["Deep Neural Network"], target_names=CLASSES, zero_division=0))

    # Confusion matrices
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (name, p) in zip(ax, preds.items()):
        ConfusionMatrixDisplay.from_predictions(y_te, p, display_labels=CLASSES, ax=a,
                                                cmap="Blues", colorbar=False)
        a.set_title(name)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "07_confusion_matrices.png", dpi=130); plt.close(fig)

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot.bar(ax=ax, rot=0)
    ax.set(title="DNN vs traditional ML (test set)", ylim=(0, 1.05)); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "08_model_comparison.png", dpi=130); plt.close(fig)

    # Save artefacts for the Streamlit app
    model.save(MODEL_DIR / "dnn_model.keras")
    joblib.dump(pre, MODEL_DIR / "preprocessor.joblib")
    info = {"classes": CLASSES, "groups": FEATURE_GROUPS, "features": feature_metadata(X_tr), "best_params": best,
            "subject": SUBJECT, "n_train": len(X_tr), "test_metrics": table.to_dict("records")}
    (MODEL_DIR / "feature_info.json").write_text(json.dumps(info, indent=2))
    print("\nDone. Run the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
