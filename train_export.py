"""Run ONCE (python train_export.py). Reproduces the notebook's final pipeline
(same split, scaler and models) and exports everything the website shows:
artifacts/{scaler,rf,lr}.joblib and artifacts/metrics.json (real values only)."""
import json, os, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, roc_curve, make_scorer)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

CSV = os.environ.get("PERSONALITY_CSV", "personality_datasert.csv")
df = pd.read_csv(CSV)

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Convert categorical columns into numerical values
df["Stage_fear"] = df["Stage_fear"].map({"No": 0, "Yes": 1})
df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"No": 0, "Yes": 1})
df["Personality"] = df["Personality"].map({"Introvert": 0, "Extrovert": 1})
assert df.isnull().sum().sum() == 0, "Missing values found"

X, y = df.drop("Personality", axis=1), df["Personality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27, stratify=y)
scaler = StandardScaler().fit(X_train)          # fitted on DataFrame -> keeps feature names
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

models = {
    "Naive Bayes": GaussianNB(), "Logistic Regression": LogisticRegression(max_iter=1000),
    "Ridge Classifier": RidgeClassifier(), "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K Nearest Neighbor": KNeighborsClassifier(), "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
}

# 10-fold CV comparison (as in notebook; cv defined BEFORE use)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {"Accuracy": "accuracy", "AUC": "roc_auc", "Recall": "recall", "Precision": "precision",
           "F1": "f1", "Kappa": make_scorer(cohen_kappa_score), "MCC": make_scorer(matthews_corrcoef)}
cv_rows = []
for name, clf in models.items():
    try:
        s = cross_validate(clf, X, y, cv=cv, scoring=scoring, error_score=np.nan)
    except ValueError:
        print("Skipped (failed every fold):", name); continue
    ok = int((~np.isnan(s["test_Accuracy"])).sum())   # folds that fitted successfully
    if ok == 0:
        print("Skipped (failed every fold):", name); continue
    if ok < 10:
        print(f"Note: {name} fitted on only {ok}/10 folds (singular covariance); averaged over those.")
    cv_rows.append({"model": name, "folds_ok": ok,
                    **{k.lower(): float(np.nanmean(s[f"test_{k}"])) for k in scoring}})
cv_rows.sort(key=lambda r: r["accuracy"], reverse=True)

# Test-set evaluation of the two focus models
test = {}
for name in ["Random Forest", "Logistic Regression"]:
    m = models[name].fit(Xtr, y_train)
    pred, prob = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    test[name] = {
        "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob), "mcc": matthews_corrcoef(y_test, pred),
        "kappa": cohen_kappa_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),   # [[TN, FP],[FN, TP]]
        "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
    }

rf = models["Random Forest"]
imp = sorted(zip(X.columns, rf.feature_importances_), key=lambda t: t[1], reverse=True)
out = {"features": list(X.columns), "n_train": len(X_train), "n_test": len(X_test),
       "test": test, "cv_comparison": cv_rows,
       "feature_importance": [{"feature": f, "importance": float(v)} for f, v in imp]}

os.makedirs("artifacts", exist_ok=True)
joblib.dump(scaler, "artifacts/scaler.joblib")
joblib.dump(rf, "artifacts/rf.joblib")
joblib.dump(models["Logistic Regression"], "artifacts/lr.joblib")
json.dump(out, open("artifacts/metrics.json", "w"), indent=1, default=float)
print("Exported. RF test accuracy:", round(test["Random Forest"]["accuracy"], 4),
      "| LR:", round(test["Logistic Regression"]["accuracy"], 4))
