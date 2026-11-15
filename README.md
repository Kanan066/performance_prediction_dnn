# Student Academic Performance Prediction using Deep Neural Networks

Predicts a student's final performance category (**Low** <10, **Average** 10–14, **High** 15–20, based on grade G3)
from demographic, academic, social and behavioural data, using the UCI Student Performance dataset.

## How to run
```bash
pip install -r requirements.txt     # Python 3.10 - 3.12 recommended (TensorFlow support)
python train.py                     # downloads data, runs the full pipeline, saves model + plots
streamlit run app.py                # launches the web app
```
If the automatic download fails (no internet / UCI down), download the dataset from the link in the brief
and place `student-por.csv` (Portuguese, 649 rows) in the `data/` folder, then run `python train.py` again.
To use the Mathematics dataset instead, set `SUBJECT = "mat"` at the top of `train.py`.

## Brief requirement → where it is done
| Requirement | Location |
|---|---|
| Data preprocessing | `preprocess()`, `build_preprocessor()` (scaling + one-hot encoding) in `train.py` |
| Exploratory data analysis | `run_eda()` → `outputs/plots/01–04` |
| Feature selection | 20 chosen features in `FEATURE_GROUPS`; `select_features()` ranks all attributes by mutual information (training data only) → plot `05` + `outputs/feature_mutual_information.csv` |
| DNN architecture design | `build_dnn()` (Dense + BatchNorm + Dropout, softmax output) |
| Model training | `train_dnn()` (early stopping, LR scheduling, class weights) → plot `06` |
| Hyperparameter tuning | `tune_dnn()` random search → `outputs/hyperparameter_search.csv` |
| Performance evaluation | accuracy, precision, recall, macro-F1, confusion matrix on held-out test set |
| Comparison with traditional ML | Random Forest + Logistic Regression → `outputs/model_comparison.csv`, plots `07–08` |
| Streamlit web app | `app.py` (inputs grouped: Demographic, Academic, Social, Behavioural) |

## Design decisions (useful for the interview)
- **Classification, not regression:** the app must display a performance *category*; G3 is binned into 3 classes.
- **70/15/15 stratified split:** validation set is used for early stopping and tuning; the test set is only touched once at the end.
- **No data leakage:** feature selection and the scaler/encoder are fitted on the training split only.
- **Class imbalance:** "High" is the minority class, so class weights and macro-averaged metrics are used.
- **G1 / G2 (earlier period grades)** are strong predictors and are kept as input attributes, as in the dataset description. Predictions are most accurate when they are provided.
- **Why compare with Random Forest?** On small tabular data (~650 rows) traditional models are often as good as a DNN; the comparison reports this honestly.

## Project structure
```
train.py            full ML/DL pipeline
app.py              Streamlit web app
.streamlit/         dark theme config
requirements.txt
data/               dataset CSV (auto-downloaded)
model/              dnn_model.keras, preprocessor.joblib, feature_info.json   (created by train.py)
outputs/            plots + metric tables                                      (created by train.py)
```

## Features used (20)
- **Demographic:** address, Medu, Mjob, Fjob, guardian
- **Academic:** G1, G2, failures, schoolsup, higher
- **Social:** famrel, freetime, goout, activities, internet
- **Behavioural:** studytime, absences, Dalc, Walc, health
