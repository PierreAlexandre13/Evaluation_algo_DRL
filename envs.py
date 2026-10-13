"""Création des environnements (entraînement, évaluation, rendu vidéo)."""
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def make_train_env(env_id, n_envs, seed, normalize=False, gamma=0.99, monitor_dir=None):
    """Environnement vectorisé d'entraînement (avec Monitor, et VecNormalize optionnel)."""
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=vec_cls,
                       monitor_dir=monitor_dir)
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=gamma)
    return env


def make_eval_env(env_id, seed, normalize=False, vecnorm_path=None, render=False):
    """Environnement d'évaluation (1 env, récompenses brutes).

    - normalize=True        : VecNormalize vierge (les stats sont synchronisées par le callback)
    - vecnorm_path=<fichier>: charge les statistiques sauvegardées à l'entraînement
    - render=True           : render_mode='rgb_array' pour enregistrer des vidéos
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array" if render else None)
        return Monitor(env)

    venv = DummyVecEnv([_init])
    venv.seed(seed)
    if vecnorm_path is not None and Path(vecnorm_path).exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
    elif normalize:
        venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)
    return venv
