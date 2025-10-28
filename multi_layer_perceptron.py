# ===== MLP (same style as LR) =====
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.base import clone
from ucimlrepo import fetch_ucirepo

# ---------- 1) Preprocess ----------
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

# ---------- 2) MLP base ----------
def make_mlp_base() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256,128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,             # L2 regularization for MLP
        learning_rate_init=1e-3,
        batch_size=128,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    )

# ---------- 3) Pipeline ----------
def make_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("prep", preprocess),
        ("clf", make_mlp_base()),
    ])

# ---------- 4)grid search parameter ----------
def make_param_grid() -> List[Dict[str, Any]]:
    return [
        {
            "clf__hidden_layer_sizes": [(64,), (128, 64)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [1e-5, 1e-4, 1e-3],
            "clf__learning_rate_init": [ 1e-5, 1e-4, 1e-3],
            "clf__batch_size": [32,64, 128],
        }
    ]

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

# ---------- 6) Plot Confusion Matrix ----------
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix (MLP)")
    plt.tight_layout()
    plt.show()

# ---------- 7) Plot Learning Curve ----------
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
    plt.fill_between(sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
    plt.plot(sizes, valid_mean, label="CV score")
    plt.fill_between(sizes, valid_mean-valid_std, valid_mean+valid_std, alpha=0.2)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    plt.title("Learning Curve (MLP)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ---------- 8) Plot: ROC ----------
def plot_roc(model, X_test, y_test):
    if len(np.unique(y_test)) != 2:
        print("[Info] Dataset is not binary. ROC skipped.")
        return
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test)).ravel()
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    auc_value = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (MLP)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# 1) fetch data
dataset = fetch_ucirepo(id=891)   
# dataset 1 id=697
# dataset 2 id=109
# dataset 3 id=848
# dataset 4 id=891
# 2) separate features and  targets
X = dataset.data.features
y = dataset.data.targets
y = y.squeeze("columns") if hasattr(y, "columns") else pd.Series(y)
mask = y.notna()
if mask.sum() < len(y):
    print(f"[Info] Dropping {len(y) - mask.sum()} rows with NaN labels.")
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)


le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# 3)construct pipeline and tune parameters
preprocess, _, _ = make_preprocess(X)
pipe = make_pipeline(preprocess)
best_model, grid = tune_with_cv(pipe, X_train, y_train)

# 4) result
print("\n[GridSearch] Best params:", grid.best_params_)
print("[GridSearch] Best CV balanced accuracy: {:.3f}".format(grid.best_score_))

y_pred = best_model.predict(X_test)
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred))

plot_confusion_matrix(best_model, X_test, y_test)
plot_learning_curve(best_model, X_train, y_train)
plot_roc(best_model, X_test, y_test)