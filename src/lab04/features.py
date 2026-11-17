"""Pre-cutoff behaviour available to a prediction after day 20."""

import pandas as pd


def player_features(df: pd.DataFrame, cutoff: int = 20) -> pd.DataFrame:
    observed = df.loc[df.day.between(0, cutoff)].copy()
    if observed.empty:
        raise ValueError("No sessions in observation window")
    g = observed.groupby("player_id", sort=True)
    f = pd.DataFrame(
        {
            "n_sessions": g.session_id.nunique(),
            "avg_len": g.length_s.mean(),
            "total_len": g.length_s.sum(),
            "kills_per_session": g.kills.mean(),
            "deaths_per_session": g.deaths.mean(),
            "kd_ratio": (g.kills.sum() + 1) / (g.deaths.sum() + 1),
            "explored_mean": g.explored.mean(),
            "chats_per_session": g.chats.mean(),
            "active_days": g.day.nunique(),
            "last_day": g.day.max(),
        }
    )
    f["sessions_per_active_day"] = f.n_sessions / f.active_days
    f["len_std"] = g.length_s.std().fillna(0)
    f["night_share"] = observed.assign(night=observed.hour.lt(6)).groupby("player_id").night.mean()
    centered_day = observed.day - g.day.transform("mean")
    centered_len = observed.length_s - g.length_s.transform("mean")
    num = (centered_day * centered_len).groupby(observed.player_id).sum()
    den = centered_day.pow(2).groupby(observed.player_id).sum()
    f["length_trend"] = num.div(den.replace(0, float("nan"))).fillna(0)
    early = observed[observed.day <= 10].groupby("player_id").size().reindex(f.index, fill_value=0)
    late = observed[observed.day > 10].groupby("player_id").size().reindex(f.index, fill_value=0)
    f["activity_trend"] = (late / 10 - early / 11) / ((early + late) / 21)
    f["recency"] = cutoff - f.last_day
    return f


def churn_labels(df: pd.DataFrame, index: pd.Index) -> pd.Series:
    """No return in the full 10-day future window; metadata is ignored."""
    returning = set(df.loc[df.day.between(21, 30), "player_id"])
    return pd.Series([int(pid not in returning) for pid in index], index=index, name="churned")
