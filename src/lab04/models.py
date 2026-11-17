"""Clustering and honest train/validation/test churn evaluation."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STYLE_FEATURES = ["kills_per_session", "deaths_per_session", "kd_ratio", "explored_mean", "chats_per_session"]


def cluster(X, k_range=range(2, 8), seed=42):
    Xs = StandardScaler().fit_transform(X)
    scores = {k: float(silhouette_score(Xs, KMeans(k, n_init=20, random_state=seed).fit_predict(Xs))) for k in k_range}
    best = max(scores, key=scores.get)
    labels = KMeans(best, n_init=20, random_state=seed).fit_predict(Xs)
    return best, scores, labels


def precision_top(y, probabilities, fraction=0.2):
    count = max(1, int(np.ceil(len(y) * fraction)))
    order = np.argsort(-np.asarray(probabilities), kind="stable")[:count]
    return float(np.asarray(y)[order].mean())


def evaluate_churn(X, y, seed=42):
    train, test = train_test_split(X.index, test_size=0.2, random_state=seed, stratify=y)
    train, val = train_test_split(train, test_size=0.25, random_state=seed, stratify=y.loc[train])
    models = {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1)),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=1),
    }
    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(X.loc[train], y.loc[train])
        pval = model.predict_proba(X.loc[val])[:, 1]
        rows.append(dict(model=name, validation_auc=float(roc_auc_score(y.loc[val], pval))))
    chosen = max((r for r in rows if r["model"] != "Dummy"), key=lambda r: r["validation_auc"])["model"]
    # Selection is complete before evaluating the untouched test partition.
    for row in rows:
        name = row["model"]
        p = models[name].predict_proba(X.loc[test])[:, 1]
        predictions[name] = p
        row.update(test_auc=float(roc_auc_score(y.loc[test], p)), precision_top20=precision_top(y.loc[test], p))
    p = predictions[chosen]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(500):
        sample = rng.integers(0, len(test), len(test))
        if y.loc[test].iloc[sample].nunique() == 2:
            boot.append(roc_auc_score(y.loc[test].iloc[sample], p[sample]))
    imp = permutation_importance(
        models[chosen], X.loc[test], y.loc[test], scoring="roc_auc", n_repeats=15, random_state=seed
    )
    importance = pd.DataFrame({"feature": X.columns, "mean": imp.importances_mean, "std": imp.importances_std})
    return dict(
        models=models,
        metrics=rows,
        selected=chosen,
        split={"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()},
        probabilities=predictions,
        importance=importance,
        auc_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def churn_model(X, y, seed=42):
    result = evaluate_churn(X, y, seed)
    name = result["selected"]
    row = next(r for r in result["metrics"] if r["model"] == name)
    return result["models"][name], row["test_auc"]
