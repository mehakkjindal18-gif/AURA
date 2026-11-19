
"""
Train the AURA personality prediction models and export website artifacts.

Run from the project root:
    python train_export.py

Exports:
    artifacts/scaler.joblib
    artifacts/rf.joblib
    artifacts/lr.joblib
    artifacts/metrics.json
"""

'''import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    make_scorer
)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.dummy import DummyClassifier


warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Load and clean the dataset
# ---------------------------------------------------------

CSV = os.environ.get("PERSONALITY_CSV", "personality_datasert.csv")

df = pd.read_csv(CSV)

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

# Convert categorical columns into numerical values
categorical_mappings = {
    "Stage_fear": {"No": 0, "Yes": 1},
    "Drained_after_socializing": {"No": 0, "Yes": 1},
    "Personality": {"Introvert": 0, "Extrovert": 1}
}

for column, mapping in categorical_mappings.items():
    if column not in df.columns:
        raise ValueError(f"Required column is missing: {column}")

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.title()
        .map(mapping)
    )

# Convert the remaining questionnaire features to numbers
numeric_features = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]

for column in numeric_features:
    if column not in df.columns:
        raise ValueError(f"Required column is missing: {column}")

    df[column] = pd.to_numeric(df[column], errors="coerce")

# Check for missing or unmapped values
if df.isnull().sum().any():
    raise ValueError(
        "Missing or unmapped values found:\n"
        f"{df.isnull().sum()}"
    )

# Keep the same seven input features in a fixed order
FEATURES = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency"
]

X = df[FEATURES].copy()
y = df["Personality"].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "The target must contain both classes: "
        "Introvert (0) and Extrovert (1)."
    )

print("Cleaned dataset shape:", df.shape)
print("Duplicate rows removed.")
print("Missing values:", int(df.isnull().sum().sum()))


# ---------------------------------------------------------
# 2. Split into training and testing data
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=27,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ---------------------------------------------------------
# 3. Define all 13 classification models
# ---------------------------------------------------------

models = {
    "Naive Bayes": GaussianNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Ridge Classifier": RidgeClassifier(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "K Nearest Neighbor": KNeighborsClassifier(),

    "SVM": SVC(
        probability=True
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        random_state=42
    ),

    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),

    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),

    "Dummy Classifier": DummyClassifier(
        strategy="most_frequent"
    )
}


# ---------------------------------------------------------
# 4. Compare models using 10-fold stratified CV
#    CV uses training data only.
#    Scaling is fitted separately inside every fold.
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scoring = {
    "Accuracy": "accuracy",
    "AUC": "roc_auc",
    "Recall": "recall",
    "Precision": "precision",
    "F1": "f1",
    "Kappa": make_scorer(cohen_kappa_score),
    "MCC": make_scorer(matthews_corrcoef)
}

cv_rows = []

for name, classifier in models.items():

    print(f"Cross-validating: {name}")
    start_time = time.time()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clone(classifier))
    ])

    try:
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            error_score=np.nan
        )

    except ValueError as error:
        print(f"Skipped {name}: {error}")
        continue

    elapsed_time = time.time() - start_time

    valid_folds = int(
        np.sum(~np.isnan(scores["test_Accuracy"]))
    )

    if valid_folds == 0:
        print(f"Skipped {name}: no successful folds.")
        continue

    if valid_folds < cv.get_n_splits():
        print(
            f"Note: {name} completed "
            f"{valid_folds}/{cv.get_n_splits()} folds."
        )

    cv_rows.append({
        "model": name,
        "folds_ok": valid_folds,
        "accuracy": float(np.nanmean(scores["test_Accuracy"])),
        "auc": float(np.nanmean(scores["test_AUC"])),
        "recall": float(np.nanmean(scores["test_Recall"])),
        "precision": float(np.nanmean(scores["test_Precision"])),
        "f1": float(np.nanmean(scores["test_F1"])),
        "kappa": float(np.nanmean(scores["test_Kappa"])),
        "mcc": float(np.nanmean(scores["test_MCC"])),
        "time_seconds": float(elapsed_time)
    })

cv_rows.sort(
    key=lambda row: row["accuracy"],
    reverse=True
)

print("\nCross-validation comparison:")
for row in cv_rows:
    print(
        row["model"],
        "Accuracy:",
        round(row["accuracy"], 4)
    )


# ---------------------------------------------------------
# 5. Fit the final scaler on training data only
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 6. Train and evaluate the two required models
# ---------------------------------------------------------

test = {}
fitted_models = {}

for name in ["Random Forest", "Logistic Regression"]:

    model = clone(models[name])

    model.fit(
        X_train_scaled,
        y_train
    )

    fitted_models[name] = model

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    test[name] = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),

        "mcc": float(
            matthews_corrcoef(y_test, predictions)
        ),

        "kappa": float(
            cohen_kappa_score(y_test, predictions)
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        ).tolist(),

        "roc": {
            "fpr": [
                round(float(value), 4)
                for value in fpr
            ],
            "tpr": [
                round(float(value), 4)
                for value in tpr
            ]
        }
    }


# ---------------------------------------------------------
# 7. Random Forest feature importance
# ---------------------------------------------------------

rf = fitted_models["Random Forest"]

feature_importance = sorted(
    zip(FEATURES, rf.feature_importances_),
    key=lambda item: item[1],
    reverse=True
)


# ---------------------------------------------------------
# 8. Export artifacts for the existing AURA website
# ---------------------------------------------------------

metrics = {
    "features": FEATURES,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "test": test,
    "cv_comparison": cv_rows,
    "feature_importance": [
        {
            "feature": feature,
            "importance": float(importance)
        }
        for feature, importance in feature_importance
    ]
}

os.makedirs("artifacts", exist_ok=True)

joblib.dump(
    scaler,
    "artifacts/scaler.joblib"
)

joblib.dump(
    fitted_models["Random Forest"],
    "artifacts/rf.joblib"
)

joblib.dump(
    fitted_models["Logistic Regression"],
    "artifacts/lr.joblib"
)

with open("artifacts/metrics.json", "w") as file:
    json.dump(
        metrics,
        file,
        indent=2
    )


# ---------------------------------------------------------
# 9. Print final results
# ---------------------------------------------------------

print("\nArtifacts exported successfully.")

print(
    "Random Forest test accuracy:",
    round(test["Random Forest"]["accuracy"], 4)
)

print(
    "Logistic Regression test accuracy:",
    round(test["Logistic Regression"]["accuracy"], 4)
)

print("\nSaved files:")
print("artifacts/scaler.joblib")
print("artifacts/rf.joblib")
print("artifacts/lr.joblib")
print("artifacts/metrics.json")'''


"""
Train the AURA personality prediction models and export website artifacts.

Run from the project root:
    python train_export.py

Exports:
    artifacts/scaler.joblib
    artifacts/rf.joblib
    artifacts/lr.joblib
    artifacts/metrics.json
"""

"""
Train the AURA personality prediction models and export the website artifacts.

Run from the project root:
    python train_export.py

Pipeline
  1. Load the CSV, REMOVE exact duplicate rows, encode Yes/No answers and the label.
  2. 80/20 stratified split (random_state=27). The test set is used only once, at the end.
  3. Compare 13 classifiers with 10-fold stratified cross-validation on the TRAINING data only
     (the scaler is fitted inside each fold, so nothing leaks).
  4. Tune Random Forest and Logistic Regression with GridSearchCV (training data only).
  5. Evaluate the default and tuned versions of the two required models once on the test set.
  6. Export scaler, tuned models, and metrics.json for the Flask website.

Exports: artifacts/scaler.joblib, artifacts/rf.joblib, artifacts/lr.joblib, artifacts/metrics.json
"""
import json, os, time, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             matthews_corrcoef, cohen_kappa_score, confusion_matrix, roc_curve, make_scorer)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.environ.get("PERSONALITY_CSV", os.path.join(BASE, "personality_datasert.csv"))

# 1. Load, remove duplicates, encode -------------------------------------------------------------
raw = pd.read_csv(CSV)
n_raw = len(raw)
df = raw.drop_duplicates().copy()
n_dup = n_raw - len(df)
for col, mp in {"Stage_fear": {"No": 0, "Yes": 1}, "Drained_after_socializing": {"No": 0, "Yes": 1},
                "Personality": {"Introvert": 0, "Extrovert": 1}}.items():
    df[col] = df[col].astype("string").str.strip().str.title().map(mp)
FEATURES = ["Time_spent_Alone", "Stage_fear", "Social_event_attendance", "Going_outside",
            "Drained_after_socializing", "Friends_circle_size", "Post_frequency"]
if df[FEATURES + ["Personality"]].isnull().any().any():
    raise ValueError("Missing or unmapped values found.")
X, y = df[FEATURES].astype(float), df["Personality"].astype(int)
print(f"Raw rows {n_raw}, duplicates removed {n_dup}, clean rows {len(df)}")

# 2. Split ---------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27, stratify=y)
scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

# 3. Compare 13 models (10-fold stratified CV on training data) -----------------------------------
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Ridge Classifier": RidgeClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K Nearest Neighbor": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    # The default QDA cannot be fitted here: among Extroverts the two Yes/No answers are identical
    # in every row, so the class covariance matrix is singular. A small regularisation fixes it.
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {"Accuracy": "accuracy", "AUC": "roc_auc", "Recall": "recall", "Precision": "precision",
           "F1": "f1", "Kappa": make_scorer(cohen_kappa_score), "MCC": make_scorer(matthews_corrcoef)}
cv_rows = []
for name, clf in models.items():
    t0 = time.time()
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])
    s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
    cv_rows.append({"model": name, "accuracy": float(s["test_Accuracy"].mean()),
                    "accuracy_std": float(s["test_Accuracy"].std()),
                    "auc": float(s["test_AUC"].mean()), "recall": float(s["test_Recall"].mean()),
                    "precision": float(s["test_Precision"].mean()), "f1": float(s["test_F1"].mean()),
                    "kappa": float(s["test_Kappa"].mean()), "mcc": float(s["test_MCC"].mean()),
                    "time_seconds": float(time.time() - t0)})
    print(f"CV  {name:32s} {cv_rows[-1]['accuracy']:.4f}")
cv_rows.sort(key=lambda r: r["accuracy"], reverse=True)

# 4. Tune the two required models (training data only) -------------------------------------------
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       {"n_estimators": [100, 300], "max_depth": [3, 5, 8, None],
                        "min_samples_leaf": [1, 3, 5, 10], "max_features": ["sqrt", None]},
                       cv=cv, scoring="accuracy", n_jobs=-1).fit(Xs_train, y_train)
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10, 100]},
                       cv=cv, scoring="accuracy").fit(Xs_train, y_train)
print("Best RF:", rf_grid.best_params_, round(rf_grid.best_score_, 4))
print("Best LR:", lr_grid.best_params_, round(lr_grid.best_score_, 4))

# 5. Evaluate on the held-out test set -----------------------------------------------------------
def evaluate(model):
    pred, prob = model.predict(Xs_test), model.predict_proba(Xs_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    return {"accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)), "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)), "roc_auc": float(roc_auc_score(y_test, prob)),
            "mcc": float(matthews_corrcoef(y_test, pred)), "kappa": float(cohen_kappa_score(y_test, pred)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]}}

rf_default = clone(models["Random Forest"]).fit(Xs_train, y_train)
lr_default = clone(models["Logistic Regression"]).fit(Xs_train, y_train)
rf, lr = rf_grid.best_estimator_, lr_grid.best_estimator_
test = {"Random Forest": evaluate(rf), "Logistic Regression": evaluate(lr)}
test_default = {"Random Forest": evaluate(rf_default), "Logistic Regression": evaluate(lr_default)}
for k in test:
    print(f"TEST {k}: default {test_default[k]['accuracy']:.4f} -> tuned {test[k]['accuracy']:.4f}")

# 6. Export ----------------------------------------------------------------------------------------
fi = sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: t[1], reverse=True)
means = df.groupby("Personality")[FEATURES].mean()
metrics = {
    "features": FEATURES,
    "dataset": {"raw_rows": n_raw, "duplicates_removed": n_dup, "clean_rows": len(df),
                "introvert": int((y == 0).sum()), "extrovert": int((y == 1).sum())},
    "n_train": len(X_train), "n_test": len(X_test),
    "test": test,                      # tuned Random Forest and tuned Logistic Regression (final models)
    "test_default": test_default,      # default settings, for comparison
    "tuning": {"Random Forest": {"best_params": rf_grid.best_params_, "cv_accuracy": float(rf_grid.best_score_)},
               "Logistic Regression": {"best_params": lr_grid.best_params_, "cv_accuracy": float(lr_grid.best_score_)}},
    "cv_comparison": cv_rows,          # default settings, 10-fold CV on the training split
    "feature_importance": [{"feature": f, "importance": float(v)} for f, v in fi],
    "group_means": {("Introvert" if k == 0 else "Extrovert"): {c: float(v) for c, v in row.items()}
                    for k, row in means.iterrows()},
}
os.makedirs(os.path.join(BASE, "artifacts"), exist_ok=True)
joblib.dump(scaler, os.path.join(BASE, "artifacts", "scaler.joblib"))
joblib.dump(rf, os.path.join(BASE, "artifacts", "rf.joblib"))
joblib.dump(lr, os.path.join(BASE, "artifacts", "lr.joblib"))
with open(os.path.join(BASE, "artifacts", "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Artifacts exported.")
