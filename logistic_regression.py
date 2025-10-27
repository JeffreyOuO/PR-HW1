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
            "clf__solver": ["lbfgs"],
            "clf__penalty": ["l2"],
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

#1 fetch data
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697 )

#2 separate features and  targets
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets
y = y.squeeze("columns")  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3 construct pipeline and tune parameters
preprocess, _, _ = make_preprocess(X)
pipe = make_pipeline(preprocess)
best_model, grid = tune_with_cv(pipe, X_train, y_train)

print("\n[GridSearch] Best params:", grid.best_params_)
print("[GridSearch] Best CV balanced accuracy: {:.3f}".format(grid.best_score_))

# 4 result
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

# ========= 比較「最佳參數」在不同比例訓練集的表現 =========
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.base import clone

# 固定測試集 (X_test, y_test) 不變，只縮小/放大訓練集的比例
train_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
rows = []

for r in train_ratios:
    # 從原本的 X_train / y_train 中，分層抽樣出 r 比例作為此次訓練資料
    if r < 1.0:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size=r, stratify=y_train, random_state=42
        )
    else:
        X_sub, y_sub = X_train, y_train

    # 用「最佳參數」建立同型模型並重新訓練
    model_r = clone(best_model)
    model_r.fit(X_sub, y_sub)

    # 在同一個測試集上評估
    y_pred_r = model_r.predict(X_test)
    rows.append({
        "train_ratio": r,
        "accuracy": accuracy_score(y_test, y_pred_r),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred_r),
        "macro_f1": f1_score(y_test, y_pred_r, average="macro"),
    })

# 輸出成績表
results_df = pd.DataFrame(rows).sort_values("train_ratio")
print("\n=== Best-Params under Different Training Ratios ===")
print(results_df.to_string(index=False))

# （可選）畫成曲線圖
plt.figure()
plt.plot(results_df["train_ratio"], results_df["accuracy"], marker="o", label="Accuracy")
plt.plot(results_df["train_ratio"], results_df["balanced_acc"], marker="o", label="Balanced Acc")
plt.plot(results_df["train_ratio"], results_df["macro_f1"], marker="o", label="Macro-F1")
plt.xlabel("Training Ratio (of original training set)")
plt.ylabel("Score")
plt.title("Effect of Training Set Size (Best Params)")
plt.legend()
plt.tight_layout()
plt.show()

# ======= Select Top-10 important features & retrain with best params =======
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# 1) 取得展開後的特徵名稱與係數重要度（多類別用 L2 norm 彙總）
prep = best_model.named_steps["prep"]
clf  = best_model.named_steps["clf"]

feature_names = prep.get_feature_names_out()
importance = np.sqrt((clf.coef_ ** 2).sum(axis=0))  # shape: (n_features,)

topk = 10
top_idx = importance.argsort()[::-1][:topk]
top_features = feature_names[top_idx]
top_importance = importance[top_idx]

print("\n=== Top-10 features used for retraining ===")
for name, score in zip(top_features, top_importance):
    print(f"{name:>40s} : {score:.6f}")

# 2) 用「已訓練好的前處理器」只做 transform（避免重算，且不洩漏）
def _to_dense(Xm):
    return Xm.toarray() if hasattr(Xm, "toarray") else Xm

Xtr_trans = _to_dense(prep.transform(X_train))
Xte_trans = _to_dense(prep.transform(X_test))

# 3) 只取 Top-10 欄位來訓練/測試
Xtr_top = pd.DataFrame(Xtr_trans[:, top_idx], columns=top_features, index=getattr(X_train, "index", None))
Xte_top = pd.DataFrame(Xte_trans[:, top_idx], columns=top_features, index=getattr(X_test, "index", None))

# 4) 用「相同最佳參數」建立新模型並訓練（只差在特徵變少）
best_C = clf.C
model_top10 = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    C=best_C,
    max_iter=1000,
)
model_top10.fit(Xtr_top, y_train)

# 5) 在相同測試集上評估
y_pred_top = model_top10.predict(Xte_top)

print("\n=== Performance with Top-10 features (best params) ===")
print("Accuracy         :", accuracy_score(y_test, y_pred_top))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_top))
print("Macro F1         :", f1_score(y_test, y_pred_top, average="macro"))
