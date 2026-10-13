"""Entraînement d'un run (env, algo, seed)."""
import json
import time

from algos import build_model
from callbacks import EvalLoggerCallback
from config import (ENVS, EVAL_FREQ, N_EVAL_EPISODES, TOTAL_TIMESTEPS,
                    get_hyperparams, run_dir)
from envs import make_eval_env, make_train_env
from stable_baselines3.common.utils import set_random_seed


def train_one(env_key, algo, seed, total_timesteps=None, eval_freq=None,
              force=False, device="auto"):
    path = run_dir(env_key, algo, seed)
    if (path / "final_model.zip").exists() and not force:
        print(f"[skip] {env_key}/{algo}/seed_{seed} déjà entraîné (utiliser --force pour refaire)")
        return
    path.mkdir(parents=True, exist_ok=True)

    hp = get_hyperparams(algo, env_key)
    env_id = ENVS[env_key]
    total = total_timesteps or TOTAL_TIMESTEPS[env_key]
    eval_freq = eval_freq or EVAL_FREQ
    tag = f"{env_key}/{algo}/s{seed}"
    print(f"\n=== Entraînement {tag} : {total:,} pas ===")

    set_random_seed(seed)
    train_env = make_train_env(env_id, hp["n_envs"], seed, normalize=hp["normalize"],
                               gamma=hp["kwargs"].get("gamma", 0.99),
                               monitor_dir=str(path / "monitor"))
    eval_env = make_eval_env(env_id, seed + 10_000, normalize=hp["normalize"])

    model = build_model(algo, train_env, seed, hp, device=device)
    callback = EvalLoggerCallback(eval_env, path, eval_freq, N_EVAL_EPISODES, tag=tag)

    t0 = time.time()
    model.learn(total_timesteps=total, callback=callback)
    total_time = time.time() - t0

    model.save(path / "final_model")
    if hp["normalize"]:
        train_env.save(str(path / "vecnormalize.pkl"))

    with open(path / "info.json", "w") as f:
        json.dump(dict(env=env_key, env_id=env_id, algo=algo, seed=seed,
                       total_timesteps=total, total_time_s=total_time,
                       eval_time_s=callback.eval_time, hyperparams=hp),
                  f, indent=2, default=str)

    train_env.close()
    eval_env.close()
    print(f"=== Terminé {tag} en {total_time/60:.1f} min ===")
