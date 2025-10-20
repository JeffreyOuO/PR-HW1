# ===== mlp_classifier.py =====
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
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


# ---------- 2) MLP base ----------
def make_mlp_base() -> MLPClassifier:
    # hidden_layer_sizes 可調整；early_stopping=True 可自動提前停止
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,               # L2 正則
        learning_rate_init=1e-3,
        max_iter=300,
        batch_size=128,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    )


# ---------- 3) construct Pipeline ----------
def make_pipeline(preprocess: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("prep", preprocess),
        ("clf", make_mlp_base()),
    ])


# ---------- 4) grid search parameter ----------
def make_param_grid() -> List[Dict[str, Any]]:
    return [
        {
            "clf__hidden_layer_sizes": [(64,), (128, 64)],
            "clf__alpha": [1e-5, 1e-4, 1e-3],
            "clf__learning_rate_init": [3e-4, 1e-3, 3e-3],
            "clf__batch_size": [64, 128, 256],
        }
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# fetch data
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets.squeeze("columns")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y)   # y 是原本的字串 Series
# 後面都用 y_enc
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# construct pipeline & tune
preprocess, _, _ = make_preprocess(X)
pipe = make_pipeline(preprocess)
best_model, grid = tune_with_cv(pipe, X_train, y_train)

print("\n[GridSearch] Best params:", grid.best_params_)
print("[GridSearch] Best CV balanced accuracy: {:.3f}".format(grid.best_score_))

# result
y_pred = best_model.predict(X_test)
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix (Best MLP on Test)")
plt.tight_layout()
plt.show()
