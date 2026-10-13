"""Construction des modèles Stable-Baselines3."""
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


def build_model(algo: str, env, seed: int, hp: dict, device: str = "auto"):
    kwargs = dict(hp["kwargs"])
    if algo == "TD3":
        n_actions = env.action_space.shape[-1]
        sigma = hp.get("action_noise_sigma", 0.1)
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions)
        )
    return ALGO_CLASSES[algo]("MlpPolicy", env, seed=seed, device=device, verbose=0, **kwargs)
