"""Point d'entrée : entraînement, test final, vidéos, métriques et graphiques.

Exemples :
    python main.py train    --envs Walker2d --algos PPO SAC TD3 --seeds 0 1 2
    python main.py evaluate
    python main.py video
    python main.py plot
    python main.py all --quick          # test rapide de toute la chaîne
"""
import argparse

from config import (ALGOS, ENVS, FIGURES_DIR, FINAL_TEST_CSV, N_FINAL_TEST_EPISODES,
                    RESULTS_DIR, SEEDS)


def parse_args():
    p = argparse.ArgumentParser(description="Comparaison PPO / SAC / TD3 (Gymnasium + SB3)")
    p.add_argument("command", choices=["train", "evaluate", "video", "plot", "all"])
    p.add_argument("--envs", nargs="+", default=list(ENVS), choices=list(ENVS))
    p.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--timesteps", type=int, default=None, help="surcharge du budget d'entraînement")
    p.add_argument("--eval-freq", type=int, default=None)
    p.add_argument("--force", action="store_true", help="réentraîner même si le run existe")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--which", choices=["final", "best"], default="final",
                   help="modèle utilisé pour le test et les vidéos")
    p.add_argument("--n-test-episodes", type=int, default=N_FINAL_TEST_EPISODES)
    p.add_argument("--video-episodes", type=int, default=2)
    p.add_argument("--quick", action="store_true", help="mini-budget pour vérifier que tout tourne")
    args = p.parse_args()
    if args.quick:
        args.timesteps = args.timesteps or 30_000
        args.eval_freq = args.eval_freq or 5_000
        args.seeds = args.seeds[:1]
        args.n_test_episodes = min(args.n_test_episodes, 3)
    return args


def cmd_train(a):
    from train import train_one
    for env_key in a.envs:
        for algo in a.algos:
            for seed in a.seeds:
                train_one(env_key, algo, seed, a.timesteps, a.eval_freq, a.force, a.device)


def cmd_evaluate(a):
    from evaluate import evaluate_all
    evaluate_all(a.envs, a.algos, a.seeds, a.which, a.n_test_episodes)


def cmd_video(a):
    from videos import record_best_videos
    record_best_videos(a.envs, a.algos, a.seeds, a.which, a.video_episodes)


def cmd_plot(a):
    import pandas as pd
    from metrics import compute_run_metrics, load_eval_logs, summarize
    from plots import plot_learning_curves, plot_summary

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logs = load_eval_logs(RESULTS_DIR, a.envs, a.algos, a.seeds)
    run_metrics = compute_run_metrics(logs)

    if FINAL_TEST_CSV.exists():
        test = pd.read_csv(FINAL_TEST_CSV)
        test = test[test.which == a.which][["env", "algo", "seed", "test_reward"]]
        run_metrics = run_metrics.merge(test, on=["env", "algo", "seed"], how="left")

    summary = summarize(run_metrics)
    run_metrics.to_csv(RESULTS_DIR / "metrics_per_run.csv", index=False)
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))

    thresholds = run_metrics.groupby("env")["threshold"].first().to_dict()
    plot_learning_curves(logs, thresholds, FIGURES_DIR / "learning_curves.png")
    plot_summary(summary, FIGURES_DIR / "summary_metrics.png")
    print(f"\nFigures -> {FIGURES_DIR}/ ; tableaux -> {RESULTS_DIR}/metrics_*.csv")


def main():
    a = parse_args()
    steps = {"train": [cmd_train], "evaluate": [cmd_evaluate], "video": [cmd_video],
             "plot": [cmd_plot], "all": [cmd_train, cmd_evaluate, cmd_video, cmd_plot]}
    for fn in steps[a.command]:
        fn(a)


if __name__ == "__main__":  # indispensable : SubprocVecEnv (multiprocessing)
    main()
