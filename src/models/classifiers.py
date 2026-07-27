"""Classical classifier registry (used for the gassing/failure label)."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

RS = 42


def build_classifiers() -> Dict[str, Any]:
    clf: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RS),
        "SVC-RBF":            SVC(kernel="rbf", probability=True, random_state=RS),
        "KNN-clf":            KNeighborsClassifier(n_neighbors=5),
        "RandomForest-clf":   RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RS),
        "MLP-clf":            MLPClassifier(hidden_layer_sizes=(16,), activation="relu",
                                            solver="lbfgs", alpha=0.01, max_iter=2000,
                                            random_state=RS),
        "GaussianNB":         GaussianNB(),
    }
    try:
        from xgboost import XGBClassifier
        clf["XGBoost-clf"] = XGBClassifier(n_estimators=400, max_depth=6, n_jobs=-1,
                                            random_state=RS, verbosity=0,
                                            use_label_encoder=False,
                                            eval_metric="logloss")
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        clf["LightGBM-clf"] = LGBMClassifier(n_estimators=400, n_jobs=-1,
                                              random_state=RS, verbosity=-1)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        clf["CatBoost-clf"] = CatBoostClassifier(iterations=400, verbose=0,
                                                  random_seed=RS, thread_count=-1)
    except Exception:
        pass
    return clf


NEEDS_SCALING = {"LogisticRegression", "SVC-RBF", "KNN-clf", "MLP-clf"}
