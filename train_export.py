
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

import json
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

# Duplicates: the report uses all 2,900 rows (402 exact duplicates are kept and
# disclosed as a limitation). Set REMOVE_DUPLICATES=1 to test without them.
if os.environ.get("REMOVE_DUPLICATES") == "1":
    df = df.drop_duplicates().copy()
    print("Duplicate rows removed.")
else:
    df = df.copy()

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
#    CV is run on the full dataset (same as the notebook).
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

    pipeline = clone(classifier)

    try:
        scores = cross_validate(
            pipeline,
            X,
            y,
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
print("artifacts/metrics.json")

