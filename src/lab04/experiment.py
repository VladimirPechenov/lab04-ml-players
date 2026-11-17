"""Run all experiments and save data, figures and a machine-readable audit."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from lab04.features import churn_labels, player_features
from lab04.make_dataset import make
from lab04.models import STYLE_FEATURES, cluster, evaluate_churn

ROOT = Path(__file__).resolve().parents[2]


def run(root=ROOT):
    root = Path(root)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    df = make()
    df.to_csv(root / "data/events.csv", index=False)
    X = player_features(df)
    y = churn_labels(df, X.index)
    X.to_csv(out / "features.csv")
    y.to_csv(out / "labels.csv")
    k, scores, labels = cluster(X[STYLE_FEATURES])
    truth = df.groupby("player_id").archetype.first().loc[X.index]
    ari = adjusted_rand_score(truth, labels)
    profiles = X.assign(cluster=labels).groupby("cluster").mean()
    profiles["players"] = pd.Series(labels).value_counts().sort_index()
    profiles["share"] = profiles.players / len(X)
    names = []
    for _, row in profiles.iterrows():
        if row.kills_per_session > 6:
            names.append("Бойцы")
        elif row.explored_mean > 0.7:
            names.append("Исследователи")
        else:
            names.append("Социальные")
    profiles["name"] = names
    profiles.to_csv(out / "cluster_profiles.csv")
    pd.crosstab(pd.Series(labels, index=X.index, name="cluster"), truth).to_csv(out / "cluster_vs_archetype.csv")
    result = evaluate_churn(X, y)
    test = result["split"]["test"]
    # Deliberate leakage demonstration on exactly the same test players.
    full_last_day = df.groupby("player_id").day.max().reindex(X.index).to_frame("last_day")
    leak = LogisticRegression().fit(full_last_day.loc[result["split"]["train"]], y.loc[result["split"]["train"]])
    leaky_auc = roc_auc_score(y.loc[test], leak.predict_proba(full_last_day.loc[test])[:, 1])
    plt.rcParams.update({"figure.figsize": (9, 5), "font.size": 11})
    fig, ax = plt.subplots()
    ax.plot(list(scores), list(scores.values()), "o-", color="#2463a0")
    ax.set(xlabel="Number of clusters k", ylabel="Silhouette", title=f"Style segmentation: selected k={k}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "silhouette.png", dpi=160)
    plt.close(fig)
    scaled = StandardScaler().fit_transform(X[STYLE_FEATURES])
    pca = PCA(n_components=2).fit(scaled)
    coords = pca.transform(scaled)
    fig, ax = plt.subplots()
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=14, alpha=0.6, label=f"{c}: {profiles.loc[c, 'name']} ({mask.sum()})"
        )
    ax.set(xlabel="PC1", ylabel="PC2", title="KMeans on five standardized style features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "clusters_pca.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots()
    for name, probabilities in result["probabilities"].items():
        fpr, tpr, _ = roc_curve(y.loc[test], probabilities)
        ax.plot(fpr, tpr, label=f"{name}: {roc_auc_score(y.loc[test], probabilities):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out test ROC (300 players)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=160)
    plt.close(fig)
    importance = result["importance"].sort_values("mean")
    importance.to_csv(out / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance.feature, importance["mean"], xerr=importance["std"], color="#2463a0")
    ax.set(xlabel="ROC-AUC decrease (mean +/- SD, 15 shuffles)", title=f"Permutation importance: {result['selected']}")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(result["probabilities"], index=test).assign(churned=y.loc[test]).to_csv(out / "test_predictions.csv")
    audit = dict(
        seed=42,
        players=len(X),
        sessions=len(df),
        features=list(X.columns),
        observation_days=[0, 20],
        target_days=[21, 30],
        churn_rate=float(y.mean()),
        k=k,
        silhouette=scores,
        ari=float(ari),
        pca_variance=pca.explained_variance_ratio_.tolist(),
        selected=result["selected"],
        metrics=result["metrics"],
        auc_ci95=result["auc_ci95"],
        leaky_auc=float(leaky_auc),
        split=result["split"],
        versions={p: version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
        python=platform.python_version(),
    )
    (out / "metrics.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


if __name__ == "__main__":
    audit = run()
    print(json.dumps({k: v for k, v in audit.items() if k != "split"}, indent=2, ensure_ascii=False))
