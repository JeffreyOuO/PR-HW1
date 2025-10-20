# ===== models.py (or put in your logistic_regression.py) =====
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# ---------- 1) preprocess ----------
def make_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocess, num_cols, cat_cols

# ---------- 2)logistic regression ----------
def make_lr_base() -> LogisticRegression:
    # sklearn 1.5+ 會自動用 multinomial；class_weight 平衡不均衡類別
    return LogisticRegression(
        solver="lbfgs",
        max_iter=4000,
        class_weight="balanced",
    )

# ---------- 3) construct Pipeline ----------
def make_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("prep", preprocess),
        ("clf", make_lr_base()),
    ])

# ---------- 4) grid search parameter ----------
def make_param_grid() -> List[Dict[str, Any]]:
    return [
        {
            "clf__solver": ["lbfgs", "newton-cg", "sag"],
            "clf__penalty": ["l2"],
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": ["balanced"],  
        },
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": ["balanced"],
        },
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["elasticnet"],
            "clf__l1_ratio": [0.0, 0.5, 1.0],   
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": ["balanced"],
        },
    ]

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


from ucimlrepo import fetch_ucirepo

# fetch data
predict_students_dropout_and_academic_success = fetch_ucirepo(id=53)

# separate features and  targets
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets
y = y.squeeze("columns")  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3)construct pipeline and tune parameters
preprocess, _, _ = make_preprocess(X)
pipe = make_pipeline(preprocess)
best_model, grid = tune_with_cv(pipe, X_train, y_train)

print("\n[GridSearch] Best params:", grid.best_params_)
print("[GridSearch] Best CV balanced accuracy: {:.3f}".format(grid.best_score_))

# 4) result
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = best_model.predict(X_test)
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=y.unique())
ConfusionMatrixDisplay(cm, display_labels=y.unique()).plot(
    cmap="Blues", xticks_rotation=45
)
plt.title("Confusion Matrix (Best LR on Test)")
plt.tight_layout()
plt.show()