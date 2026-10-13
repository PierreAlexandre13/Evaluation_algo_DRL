# Comparaison PPO / SAC / TD3 — BipedalWalker, Walker2d, Humanoid

## Installation
```bash
python3 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install swig                  # requis pour compiler box2d
pip install -r requirements.txt
# Linux sans écran (rendu MuJoCo pour les vidéos) :
export MUJOCO_GL=egl              # ou osmesa
```

## Utilisation
```bash
python3 main.py all --quick                 # vérifie toute la chaîne en quelques minutes
python3 main.py train --envs Walker2d --algos PPO SAC TD3 --seeds 0 1 2
python3 main.py train                       # tout : 3 envs x 3 algos x 3 seeds (long !)
python3 main.py evaluate                    # test final (20 épisodes) -> results/final_test.csv
python3 main.py video                       # 1 vidéo par (env, algo), meilleure seed -> videos/
python3 main.py plot                        # métriques + figures -> results/ et figures/
```
Options utiles : `--timesteps N`, `--eval-freq N`, `--device cpu|cuda`, `--which final|best`,
`--force`, `--video-episodes N`. Un run déjà terminé est ignoré (reprise possible).

## Structure
| Fichier | Rôle |
|---|---|
| `config.py` | envs, seeds, budgets, hyperparamètres, paramètres des métriques |
| `algos.py` | construction des modèles SB3 (PPO / SAC / TD3 + bruit d'action TD3) |
| `envs.py` | envs d'entraînement / évaluation / rendu (VecNormalize pour PPO) |
| `callbacks.py` | évaluation périodique -> `eval_log.csv` (pas, temps d'entraînement, récompense) |
| `train.py` | entraîne un run (env, algo, seed) |
| `evaluate.py` | test final sur N épisodes |
| `videos.py` | enregistre des .mp4 |
| `metrics.py` | calcule les 4 critères |
| `plots.py` | courbes d'apprentissage + barres comparatives |
| `main.py` | CLI |

## Sorties
```
results/<env>/<algo>/seed_<k>/  eval_log.csv, final_model.zip, best_model.zip, vecnormalize.pkl, info.json
results/final_test.csv   results/metrics_per_run.csv   results/metrics_summary.csv
figures/learning_curves.png      figures/summary_metrics.png
videos/<env>/<algo>_seed<k>_final.mp4
```

## Définition des critères (voir `metrics.py`)
- **Récompense moyenne** : `final_reward` (moyenne des derniers 20 % d'évaluations) et `test_reward` (test final).
- **Efficacité en échantillons** : `auc_reward` (récompense moyenne sur tout l'entraînement) et
  `steps_to_threshold` (pas pour atteindre le seuil).
- **Vitesse temporelle de convergence** : `time_to_threshold_s` (temps d'entraînement hors évaluations).
- **Stabilité** : `within_run_std` (bruit en fin d'entraînement autour de la tendance), `max_drawdown`
  (plus grosse chute de la courbe lissée), et écart-type du `final_reward` entre seeds.
- Seuil par env = perf. initiale + 80 % de l'écart jusqu'à la meilleure perf. finale moyenne (`THRESHOLD_FRAC`).

## Points d'attention pour l'analyse
- Pour des temps comparables, lance les runs **séquentiellement sur la même machine**.
- PPO utilise 8 envs en parallèle (CPU), SAC/TD3 un seul env avec 1 mise à jour par pas : le temps
  mur dépend donc beaucoup de la machine. Précise-le dans ton rapport.
- Coût : Humanoid (2 M de pas) avec SAC/TD3 est long (plusieurs heures/seed). Commence avec
  `--timesteps` réduit. Le replay buffer (500k) de Humanoid occupe plusieurs Go de RAM.
- `--which final` (défaut) évite le biais de sélection de `best_model`, choisi sur l'env d'évaluation.
