# ===== Random Forest (same style & features as LR/MLP) =====
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, learning_curve, train_test_split
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.base import clone
from ucimlrepo import fetch_ucirepo

# ---------- 1) Preprocess ----------
# 樹模型不需要對數值特徵做標準化；保留數值原樣 + 類別做 One-Hot
def make_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocess, num_cols, cat_cols

# ---------- 2) RF base ----------
def make_rf_base() -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

# ---------- 3) Pipeline ----------
def make_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("prep", preprocess),
        ("clf", make_rf_base()),
    ])

# ---------- 4) Grid (balanced_accuracy in CV, same spirit as LR/MLP) ----------
def make_param_grid() -> List[Dict[str, Any]]:
    return [{
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt"],      # 經驗上對分類較穩
        "clf__class_weight": ["balanced"],  # 類別不平衡時更穩定
    }]

# ---------- 5) Tune with CV ----------
def tune_with_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "balanced_accuracy",
    n_splits: int = 5,
    verbose: int = 1,
    n_jobs: int = -1,
):
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

# ---------- 6) Plot: Confusion Matrix ----------
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix (Best RF)")
    plt.tight_layout()
    plt.show()

# ---------- 7) Plot: Learning Curve ----------
def plot_learning_curve(estimator, X_train, y_train, scoring="balanced_accuracy"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 8)
    est = clone(estimator)
    sizes, train_scores, valid_scores = learning_curve(
        estimator=est,
        X=X_train, y=y_train,
        cv=cv, scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1, shuffle=True, random_state=42,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    valid_mean, valid_std = valid_scores.mean(axis=1), valid_scores.std(axis=1)

    plt.figure()
    plt.plot(sizes, train_mean, label="Training score")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(sizes, valid_mean, label="CV score")
    plt.fill_between(sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    plt.title("Learning Curve (RF)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ---------- 8) Plot: ROC (2-class only) ----------
def plot_roc_if_binary(model, X_test, y_test):
    # 非二分類就跳過
    if len(np.unique(y_test)) != 2:
        print("[Info] Dataset is not binary. ROC skipped.")
        return
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test)).ravel()
    # RF 支援 predict_proba
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    auc_value = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (2-class only, RF)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ===== Data load (general naming, same as LR/MLP style) =====
# 範例：Dataset 1（可改成你要的 1~4 任一資料集 ID）
dataset = fetch_ucirepo(id=891)  # Predict Students' Dropout and Academic Success

# Separate features & target
X = dataset.data.features
y = dataset.data.targets

# ------ Clean target NaN (must do before stratified split) ------
y = y.squeeze("columns") if hasattr(y, "columns") else pd.Series(y)
mask = y.notna()
if mask.sum() < len(y):
    print(f"[Info] Dropping {len(y) - mask.sum()} rows with NaN labels.")
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

# ------ Train/test split (stratified) ------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------ Build, tune, evaluate ------
preprocess, _, _ = make_preprocess(X)
pipe = make_pipeline(preprocess)
best_model, grid = tune_with_cv(pipe, X_train, y_train)

print("\n[GridSearch] Best params:", grid.best_params_)
print("[GridSearch] Best CV balanced accuracy: {:.3f}".format(grid.best_score_))

y_pred = best_model.predict(X_test)
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred))

# ------ Plots: keep same order as LR/MLP ------
plot_confusion_matrix(best_model, X_test, y_test)
plot_learning_curve(best_model, X_train, y_train)
plot_roc_if_binary(best_model, X_test, y_test)
