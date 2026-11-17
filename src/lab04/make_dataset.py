"""Synthetic sessions. Churn means no sessions on days 21..30."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "events.csv"
ARCHETYPES = {
    "aggressive": dict(kills=9, deaths=4, explored=0.25, chats=1),
    "explorer": dict(kills=2, deaths=2, explored=0.85, chats=2),
    "social": dict(kills=3, deaths=3, explored=0.5, chats=11),
}


def make(n_players: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Latent engagement affects behaviour and return with noise.

    Eligible cohort: at least one observation session. Archetype and engagement
    are never model inputs. Future events are sampled after observation events.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        arch = rng.choice(list(ARCHETYPES))
        params = ARCHETYPES[arch]
        engagement = rng.normal()
        n_obs = 1 + rng.poisson(11 * np.exp(0.55 * engagement))
        p_return = 1 / (1 + np.exp(-(0.6 + 2.8 * engagement)))
        n_future = 1 + rng.poisson(4) if rng.random() < p_return else 0
        days = sorted(rng.integers(0, 21, n_obs).tolist())
        days += sorted(rng.integers(21, 31, n_future).tolist())
        for session, day in enumerate(days):
            length = rng.gamma(5, 100) * np.exp(0.3 * engagement)
            length *= np.exp(0.2 * engagement * day / 20)
            rows.append(
                dict(
                    player_id=f"p{pid:04d}",
                    session_id=f"p{pid:04d}_s{session:03d}",
                    day=day,
                    hour=int(rng.integers(0, 24)),
                    length_s=round(float(length), 3),
                    kills=int(rng.poisson(params["kills"])),
                    deaths=int(rng.poisson(params["deaths"])),
                    explored=round(float(np.clip(rng.normal(params["explored"], 0.08), 0, 1)), 5),
                    chats=int(rng.poisson(params["chats"])),
                    archetype=arch,
                    churned=int(n_future == 0),
                )
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    data = make()
    data.to_csv(OUT, index=False)
    print(f"{len(data)} sessions, {data.player_id.nunique()} players -> {OUT}")
