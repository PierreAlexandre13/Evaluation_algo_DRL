"""Graphiques : courbes d'apprentissage et comparaison des critères."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"PPO": "tab:blue", "SAC": "tab:orange", "TD3": "tab:green"}

# (colonne, étiquette de l'axe y)
SUMMARY_ROWS = [
    ("test_reward", "Récompense test finale"),
    ("final_reward", "Récompense finale"),
    ("auc_reward", "AUC (efficacité éch.)"),
    ("steps_to_threshold", "Pas → seuil"),
    ("time_to_threshold_s", "Temps → seuil (s)"),
    ("within_run_std", "Écart-type intra-run"),
    ("max_drawdown", "Chute max"),
]


def _mean_std_curve(g, x_key, n_points=200):
    seeds = [r for _, r in g.groupby("seed")]
    xmax = min(r[x_key].max() for r in seeds)
    grid = np.linspace(0, xmax, n_points)
    arr = np.vstack([np.interp(grid, r[x_key], r["smooth"]) for r in seeds])
    return grid, arr.mean(0), arr.std(0)


def plot_learning_curves(logs, thresholds, out_path):
    """Ligne 1 : récompense vs pas (efficacité en échantillons).
    Ligne 2 : récompense vs temps d'entraînement (vitesse temporelle)."""
    envs = sorted(logs["env"].unique())
    fig, axes = plt.subplots(2, len(envs), figsize=(5.5 * len(envs), 8), squeeze=False)
    for j, env in enumerate(envs):
        for i, (x_key, xlabel) in enumerate([("timesteps", "Pas d'environnement"),
                                             ("train_time_min", "Temps d'entraînement (min)")]):
            ax = axes[i, j]
            for algo in sorted(logs["algo"].unique()):
                g = logs[(logs.env == env) & (logs.algo == algo)]
                if g.empty:
                    continue
                x, m, s = _mean_std_curve(g, x_key)
                ax.plot(x, m, label=f"{algo} (n={g.seed.nunique()})", color=COLORS.get(algo))
                ax.fill_between(x, m - s, m + s, alpha=0.2, color=COLORS.get(algo))
            if env in thresholds:
                ax.axhline(thresholds[env], ls="--", color="gray", lw=1, label="seuil convergence")
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(env)
            if j == 0:
                ax.set_ylabel("Récompense (moyenne ± std seeds)")
            if i == 0 and j == 0:
                ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary(summary, out_path):
    """Grille critères (lignes) x environnements (colonnes), barres = algos, barres d'erreur = std seeds."""
    rows = [(c, l) for c, l in SUMMARY_ROWS if f"{c}_mean" in summary.columns]
    envs = sorted(summary["env"].unique())
    fig, axes = plt.subplots(len(rows), len(envs), figsize=(4 * len(envs), 2.7 * len(rows)),
                             squeeze=False)
    for r, (col, label) in enumerate(rows):
        for c, env in enumerate(envs):
            ax = axes[r, c]
            sub = summary[summary.env == env].sort_values("algo")
            for k, (_, row) in enumerate(sub.iterrows()):
                m, s = row[f"{col}_mean"], row.get(f"{col}_std", np.nan)
                if np.isnan(m):
                    ax.text(k, 0, "non atteint", ha="center", va="bottom", fontsize=8, rotation=90)
                    continue
                ax.bar(k, m, yerr=None if np.isnan(s) else s, capsize=4,
                       color=COLORS.get(row["algo"]))
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["algo"])
            ax.grid(axis="y", alpha=0.3)
            if r == 0:
                ax.set_title(env)
            if c == 0:
                ax.set_ylabel(label)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
