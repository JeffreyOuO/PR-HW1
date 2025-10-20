# ===== models.py (or put in your random_forest.py) =====
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# ---------- 1) preprocess ----------
def make_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),                      # 樹模型：數值特徵不需標準化
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocess, num_cols, cat_cols

# ---------- 2) random forest base ----------
def make_rf_base() -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

# ---------- 3) construct Pipeline ----------
def make_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("prep", preprocess),
        ("clf", make_rf_base()),
    ])

# ---------- 4) grid search parameter ----------
def make_param_grid():
    return [{
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt"],
        "clf__class_weight": ["balanced"]
    }]

# ---------- 5) K-fold and GridSearch ----------
def tune_with_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "balanced_accuracy",
    n_splits: int = 5,
    verbose: int = 1,
    n_jobs: int = -1,
) -> Tuple[Pipeline, GridSearchCV]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=make_param_grid(),
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid


# ===== Demo: 與你原模板相同流程（UCI 下載 → 切資料 → 調參） =====
from ucimlrepo import fetch_ucirepo

# fetch data（同原檔）
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# separate features and targets（同原檔）
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets
y = y.squeeze("columns")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# construct pipeline and tune parameters（同原檔結構）
preprocess, _, _ = make_preprocess(X)
pipe = make_pipeline(preprocess)
best_model, grid = tune_with_cv(pipe, X_train, y_train)

print("\n[GridSearch] Best params:", grid.best_params_)
print("[GridSearch] Best CV balanced accuracy: {:.3f}".format(grid.best_score_))

# result（同原檔：報表 + 混淆矩陣）
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = best_model.predict(X_test)
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=y.unique())
ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot(
    cmap="Blues", xticks_rotation=45
)
plt.title("Confusion Matrix (Best Random Forest on Test)")
plt.tight_layout()
plt.show()
