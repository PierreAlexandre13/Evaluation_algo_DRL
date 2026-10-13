"""Callback d'évaluation : journalise la courbe d'apprentissage et le temps d'entraînement."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization


class EvalLoggerCallback(BaseCallback):
    """Évalue la politique tous les `eval_freq` pas et écrit `eval_log.csv`.

    Colonnes : timesteps, train_time_s (temps d'entraînement HORS évaluation),
    wall_time_s, mean_reward, std_reward, mean_length.
    Sauvegarde aussi `best_model.zip` (+ `best_vecnormalize.pkl` si normalisation).
    """

    def __init__(self, eval_env, run_path, eval_freq, n_eval_episodes, tag="", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.run_path = Path(run_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.tag = tag
        self.rows = []
        self.best = -np.inf
        self.last_eval = 0
        self.t0 = None
        self.eval_time = 0.0

    def _on_training_start(self):
        self.t0 = time.time()
        self._evaluate()  # évaluation initiale (politique non entraînée)

    def _on_step(self):
        if self.model.num_timesteps - self.last_eval >= self.eval_freq:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.model.num_timesteps > self.last_eval:
            self._evaluate()

    def _evaluate(self):
        t_start = time.time()
        if isinstance(self.training_env, VecNormalize):
            sync_envs_normalization(self.training_env, self.eval_env)
        rewards, lengths = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
            deterministic=True, return_episode_rewards=True,
        )
        self.eval_time += time.time() - t_start

        steps = self.model.num_timesteps
        self.last_eval = steps
        mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
        self.rows.append(dict(
            timesteps=steps,
            train_time_s=time.time() - self.t0 - self.eval_time,
            wall_time_s=time.time() - self.t0,
            mean_reward=mean_r, std_reward=std_r, mean_length=float(np.mean(lengths)),
        ))
        pd.DataFrame(self.rows).to_csv(self.run_path / "eval_log.csv", index=False)

        if mean_r > self.best:
            self.best = mean_r
            self.model.save(self.run_path / "best_model")
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(str(self.run_path / "best_vecnormalize.pkl"))

        if self.verbose:
            print(f"[{self.tag}] steps={steps:>9,d}  reward={mean_r:9.1f} ± {std_r:7.1f}  "
                  f"t_train={self.rows[-1]['train_time_s']/60:6.1f} min", flush=True)
