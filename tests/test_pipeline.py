import numpy as np
import pandas as pd

from lab04.features import churn_labels, player_features
from lab04.make_dataset import make
from lab04.models import STYLE_FEATURES, cluster, evaluate_churn, precision_top


def test_future_and_metadata_cannot_change_features():
    df = make(100, seed=7)
    expected = player_features(df)
    changed = df.copy()
    changed.loc[changed.day > 20, ["kills", "length_s", "day"]] = [9999, 999999, 30]
    changed["churned"] = 1 - changed.churned
    changed["archetype"] = "unknown"
    pd.testing.assert_frame_equal(expected, player_features(changed))
    pd.testing.assert_frame_equal(expected, player_features(df.sample(frac=1, random_state=1)))
    assert np.isfinite(expected).all().all()


def test_labels_come_from_future_and_single_session_is_finite():
    df = make(80, seed=1)
    X = player_features(df)
    y = churn_labels(df, X.index)
    assert y.equals(df.groupby("player_id").churned.first().loc[X.index].rename("churned"))
    single = df.loc[(df.player_id == df.player_id.iloc[0]) & (df.day <= 20)].iloc[:1]
    f = player_features(single)
    assert f.len_std.iloc[0] == f.length_trend.iloc[0] == 0
    assert np.isfinite(f).all().all()


def test_end_to_end_and_disjoint_player_split():
    df = make(600, seed=42)
    X = player_features(df)
    y = churn_labels(df, X.index)
    k, scores, labels = cluster(X[STYLE_FEATURES])
    assert k == max(scores, key=scores.get)
    assert len(labels) == len(X) and X.shape[1] >= 10
    result = evaluate_churn(X, y)
    tr, va, te = [set(result["split"][s]) for s in ["train", "validation", "test"]]
    assert not tr & va and not tr & te and not va & te
    assert tr | va | te == set(X.index)
    chosen = next(r for r in result["metrics"] if r["model"] == result["selected"])
    assert chosen["test_auc"] >= 0.70
    assert result["models"]["Dummy"].classes_.tolist() == [0, 1]


def test_top20_precision_uses_highest_risk_players():
    assert precision_top([0, 1, 0, 0, 0], [0.1, 0.9, 0.4, 0.3, 0.2]) == 1
