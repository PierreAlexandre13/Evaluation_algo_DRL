"""Test final des politiques entraînées (récompense sur N épisodes, sans exploration)."""
import json

import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy

from algos import ALGO_CLASSES
from config import ENVS, FINAL_TEST_CSV, N_FINAL_TEST_EPISODES, get_hyperparams, run_dir
from envs import make_eval_env


def _paths(run_path, which):
    if which == "best":
        return run_path / "best_model.zip", run_path / "best_vecnormalize.pkl"
    return run_path / "final_model.zip", run_path / "vecnormalize.pkl"


def load_model_and_env(env_key, algo, seed, which="final", render=False, env_seed_offset=20_000):
    """Charge un modèle entraîné et l'environnement d'évaluation correspondant."""
    path = run_dir(env_key, algo, seed)
    model_path, vec_path = _paths(path, which)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if get_hyperparams(algo, env_key)["normalize"] and not vec_path.exists():
        raise FileNotFoundError(f"Statistiques de normalisation manquantes : {vec_path}")
    env = make_eval_env(ENVS[env_key], seed + env_seed_offset, vecnorm_path=vec_path, render=render)
    model = ALGO_CLASSES[algo].load(model_path, device="cpu")
    return model, env


def evaluate_run(env_key, algo, seed, which="final", n_episodes=N_FINAL_TEST_EPISODES):
    model, env = load_model_and_env(env_key, algo, seed, which)
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=n_episodes,
                                       deterministic=True, return_episode_rewards=True)
    env.close()
    result = dict(env=env_key, algo=algo, seed=seed, which=which,
                  test_reward=float(pd.Series(rewards).mean()),
                  test_reward_std=float(pd.Series(rewards).std(ddof=0)),
                  test_length=float(pd.Series(lengths).mean()), n_episodes=n_episodes)
    with open(run_dir(env_key, algo, seed) / f"final_test_{which}.json", "w") as f:
        json.dump(dict(result, episode_rewards=[float(r) for r in rewards]), f, indent=2)
    return result


def evaluate_all(envs, algos, seeds, which="final", n_episodes=N_FINAL_TEST_EPISODES):
    """Évalue tous les runs disponibles et met à jour results/final_test.csv."""
    rows = []
    for env_key in envs:
        for algo in algos:
            for seed in seeds:
                try:
                    res = evaluate_run(env_key, algo, seed, which, n_episodes)
                except FileNotFoundError as e:
                    print(f"[skip test] {env_key}/{algo}/seed_{seed} : {e}")
                    continue
                print(f"[test] {env_key}/{algo}/s{seed}: {res['test_reward']:.1f} ± {res['test_reward_std']:.1f}")
                rows.append(res)
    new = pd.DataFrame(rows)
    if new.empty:
        return new
    if FINAL_TEST_CSV.exists():
        old = pd.read_csv(FINAL_TEST_CSV)
        keys = new[["env", "algo", "seed"]].apply(tuple, axis=1)
        old = old[~old[["env", "algo", "seed"]].apply(tuple, axis=1).isin(set(keys))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(FINAL_TEST_CSV, index=False)
    return new
