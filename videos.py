"""Enregistrement de vidéos (.mp4) des politiques entraînées.

Sous Linux sans écran, MuJoCo a besoin d'un rendu hors-écran :
    export MUJOCO_GL=egl     # (ou osmesa)
"""
from pathlib import Path

import imageio
import pandas as pd

from config import FINAL_TEST_CSV, VIDEOS_DIR
from evaluate import load_model_and_env


def record_video(env_key, algo, seed, which="final", n_episodes=2, out_path=None, max_frames=5000):
    out_path = Path(out_path or VIDEOS_DIR / env_key / f"{algo}_seed{seed}_{which}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, env = load_model_and_env(env_key, algo, seed, which, render=True, env_seed_offset=30_000)
    fps = int(env.get_attr("metadata")[0].get("render_fps", 30))

    frames, returns = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        frames.append(env.get_images()[0])
        while len(frames) < max_frames:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            if dones[0]:
                returns.append(float(infos[0]["episode"]["r"]))
                break
            frames.append(env.get_images()[0])
    env.close()

    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    print(f"[vidéo] {out_path}  ({len(frames)} images, {fps} fps, retours = "
          f"{[round(r, 1) for r in returns]})")
    return out_path


def record_best_videos(envs, algos, seeds, which="final", n_episodes=2):
    """Pour chaque (env, algo), enregistre la vidéo de la meilleure seed (selon le test final)."""
    test = pd.read_csv(FINAL_TEST_CSV) if FINAL_TEST_CSV.exists() else None
    paths = []
    for env_key in envs:
        for algo in algos:
            seed = seeds[0]
            if test is not None:
                sub = test[(test.env == env_key) & (test.algo == algo) & (test.which == which)
                           & (test.seed.isin(seeds))]
                if not sub.empty:
                    seed = int(sub.sort_values("test_reward").iloc[-1]["seed"])
            try:
                paths.append(record_video(env_key, algo, seed, which, n_episodes))
            except FileNotFoundError as e:
                print(f"[skip vidéo] {env_key}/{algo} : {e}")
    return paths
