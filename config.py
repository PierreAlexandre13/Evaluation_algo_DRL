"""Configuration globale : environnements, algorithmes, budgets, hyperparamètres."""
from copy import deepcopy
from pathlib import Path

# --------------------------------------------------------------------------- #
# Dossiers de sortie
# --------------------------------------------------------------------------- #
RESULTS_DIR = Path("results")
VIDEOS_DIR = Path("videos")
FIGURES_DIR = Path("figures")
FINAL_TEST_CSV = RESULTS_DIR / "final_test.csv"

# --------------------------------------------------------------------------- #
# Expérience
# --------------------------------------------------------------------------- #
# clé courte (utilisée pour les dossiers) -> id Gymnasium
ENVS = {
    "BipedalWalker": "BipedalWalker-v3",
    "Walker2d": "Walker2d-v5",
    "Humanoid": "Humanoid-v5",
}
ALGOS = ["PPO", "SAC", "TD3"]
SEEDS = [0, 1, 2]

# Budget d'entraînement (en pas d'environnement) - identique pour tous les algos
TOTAL_TIMESTEPS = {
    "BipedalWalker": 1_000_000,
    "Walker2d": 1_000_000,
    "Humanoid": 2_000_000,
}

EVAL_FREQ = 10_000          # une évaluation tous les N pas d'environnement
N_EVAL_EPISODES = 5         # épisodes par évaluation pendant l'entraînement
N_FINAL_TEST_EPISODES = 20  # épisodes du test final (après entraînement)

# --------------------------------------------------------------------------- #
# Paramètres des métriques
# --------------------------------------------------------------------------- #
THRESHOLD_FRAC = 0.8   # seuil = 80 % du chemin entre perf. initiale et meilleure perf. finale
TAIL_FRAC = 0.2        # "fin d'entraînement" = derniers 20 % des évaluations
SMOOTH_WINDOW = 3      # fenêtre de lissage des courbes (en nombre d'évaluations)

# --------------------------------------------------------------------------- #
# Hyperparamètres (inspirés des valeurs usuelles de RL-Zoo / SB3)
# --------------------------------------------------------------------------- #
HYPERPARAMS = {
    "PPO": {
        "n_envs": 8,
        "normalize": True,  # VecNormalize (obs + reward) : crucial pour PPO sur MuJoCo
        "kwargs": dict(
            learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        ),
    },
    "SAC": {
        "n_envs": 1,
        "normalize": False,
        "kwargs": dict(
            learning_rate=3e-4, buffer_size=500_000, batch_size=256,
            learning_starts=10_000, tau=0.005, gamma=0.99,
            train_freq=1, gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
        ),
    },
    "TD3": {
        "n_envs": 1,
        "normalize": False,
        "action_noise_sigma": 0.1,
        "kwargs": dict(
            learning_rate=1e-3, buffer_size=500_000, batch_size=256,
            learning_starts=10_000, tau=0.005, gamma=0.99,
            train_freq=1, gradient_steps=1,
            policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
        ),
    },
}

# Surcharges par (algo, env) : seules les clés listées sont modifiées
ENV_OVERRIDES = {
    ("PPO", "BipedalWalker"): {"kwargs": dict(gamma=0.999, ent_coef=0.001, clip_range=0.18)},
}


def get_hyperparams(algo: str, env_key: str) -> dict:
    """Hyperparamètres de base + surcharges éventuelles pour (algo, env)."""
    hp = deepcopy(HYPERPARAMS[algo])
    override = ENV_OVERRIDES.get((algo, env_key), {})
    for key, value in override.items():
        if key == "kwargs":
            hp["kwargs"].update(value)
        else:
            hp[key] = value
    return hp


def run_dir(env_key: str, algo: str, seed: int) -> Path:
    return RESULTS_DIR / env_key / algo / f"seed_{seed}"
