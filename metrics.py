"""Calcul des critères d'évaluation à partir des journaux `eval_log.csv`.

Critères :
  1. Récompense moyenne  : final_reward (moyenne sur les derniers TAIL_FRAC % des évaluations)
                           + test_reward (test final, si disponible).
  2. Efficacité en échantillons :
        - auc_reward          : récompense moyenne tout au long de l'entraînement (aire sous la
                                courbe normalisée par le budget) -> plus haut = apprend plus vite
        - steps_to_threshold  : nb de pas pour atteindre le seuil (plus bas = mieux)
  3. Vitesse temporelle de convergence :
        - time_to_threshold_s : temps d'entraînement (hors évaluation) pour atteindre le seuil
  4. Stabilité :
        - within_run_std : écart-type de la courbe en fin d'entraînement autour de sa tendance
        - max_drawdown   : plus grosse chute de la courbe lissée par rapport à son maximum passé
        - écart-type du final_reward entre seeds (colonne final_reward_std du résumé)

Seuil par environnement = r_init + THRESHOLD_FRAC * (r_max - r_init), où r_init est la
récompense initiale moyenne et r_max la meilleure récompense finale moyenne parmi les algos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SMOOTH_WINDOW, TAIL_FRAC, THRESHOLD_FRAC

RUN_KEYS = ["env", "algo", "seed"]


def load_eval_logs(results_dir=RESULTS_DIR, envs=None, algos=None, seeds=None,
                   smooth_window=SMOOTH_WINDOW) -> pd.DataFrame:
    frames = []
    for p in Path(results_dir).glob("*/*/seed_*/eval_log.csv"):
        seed_dir = p.parent
        algo, env = seed_dir.parent.name, seed_dir.parent.parent.name
        seed = int(seed_dir.name.split("_")[1])
        if (envs and env not in envs) or (algos and algo not in algos) or (seeds and seed not in seeds):
            continue
        df = pd.read_csv(p)
        df["env"], df["algo"], df["seed"] = env, algo, seed
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Aucun eval_log.csv trouvé dans {results_dir}")
    logs = pd.concat(frames, ignore_index=True).sort_values(RUN_KEYS + ["timesteps"])
    logs["smooth"] = logs.groupby(RUN_KEYS)["mean_reward"].transform(
        lambda s: s.rolling(smooth_window, min_periods=1).mean())
    logs["train_time_min"] = logs["train_time_s"] / 60.0
    return logs.reset_index(drop=True)


def _auc(y, x):
    """Aire sous la courbe (trapèzes) normalisée par la largeur de x."""
    if len(x) < 2 or x[-1] == x[0]:
        return float(y[-1])
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))


def compute_run_metrics(logs, tail_frac=TAIL_FRAC, threshold_frac=THRESHOLD_FRAC) -> pd.DataFrame:
    rows = []
    for (env, algo, seed), g in logs.groupby(RUN_KEYS):
        g = g.sort_values("timesteps")
        x = g["timesteps"].to_numpy(float)
        y = g["mean_reward"].to_numpy(float)
        s = g["smooth"].to_numpy(float)

        n_tail = max(3, int(np.ceil(len(g) * tail_frac)))
        ty, tx = y[-n_tail:], x[-n_tail:]
        final = float(ty.mean())
        if len(ty) >= 3 and tx[-1] > tx[0]:
            tn = (tx - tx[0]) / (tx[-1] - tx[0])
            within = float(np.std(ty - np.polyval(np.polyfit(tn, ty, 1), tn)))
        else:
            within = float(np.std(ty))

        rows.append(dict(
            env=env, algo=algo, seed=seed,
            initial_reward=float(y[0]),
            final_reward=final,
            auc_reward=_auc(y, x),
            within_run_std=within,
            cv=within / abs(final) if final != 0 else np.nan,
            max_drawdown=float(np.max(np.maximum.accumulate(s) - s)),
            total_train_time_s=float(g["train_time_s"].iloc[-1]),
        ))
    rm = pd.DataFrame(rows)

    # Seuil de convergence par environnement
    thresholds = {}
    for env, g in rm.groupby("env"):
        r_init = g["initial_reward"].mean()
        r_max = g.groupby("algo")["final_reward"].mean().max()
        thresholds[env] = r_init + threshold_frac * (r_max - r_init)
    rm["threshold"] = rm["env"].map(thresholds)

    # Pas / temps pour atteindre le seuil (courbe lissée)
    steps, times = [], []
    for _, r in rm.iterrows():
        g = logs[(logs.env == r.env) & (logs.algo == r.algo) & (logs.seed == r.seed)]
        hit = g[g["smooth"] >= r.threshold]
        if len(hit):
            steps.append(float(hit.iloc[0]["timesteps"]))
            times.append(float(hit.iloc[0]["train_time_s"]))
        else:
            steps.append(np.nan)
            times.append(np.nan)
    rm["steps_to_threshold"] = steps
    rm["time_to_threshold_s"] = times
    return rm


def summarize(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Moyenne et écart-type sur les seeds, par (env, algo)."""
    rm = run_metrics.copy()
    rm["reached"] = rm["steps_to_threshold"].notna().astype(float)  # part des seeds ayant atteint le seuil
    candidates = ["test_reward", "final_reward", "auc_reward", "steps_to_threshold",
                  "time_to_threshold_s", "within_run_std", "cv", "max_drawdown",
                  "total_train_time_s", "reached"]
    cols = [c for c in candidates if c in rm.columns]
    agg = rm.groupby(["env", "algo"])[cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg["n_seeds"] = rm.groupby(["env", "algo"]).size()
    agg["threshold"] = rm.groupby(["env", "algo"])["threshold"].first()
    return agg.reset_index()
