# compute_metrics.py
"""
Script para cargar un modelo entrenado, evaluarlo
y calcular todas las métricas del experimento.

Uso:
    python compute_metrics.py --alg ppo --modo-eval historico
    python compute_metrics.py --alg a2c --modo-eval historico
    python compute_metrics.py --alg ql  --modo-eval historico

ACTUALMENTE EN DESUSO (DIC 2025) - ver README para más detalles.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from src.rl_algorithms.ppo_agent import PPOAgent
from src.rl_algorithms.a2c_agent import A2CAgent
from src.rl_algorithms.q_learning_agent import QLearningAgent
from src.utils.metrics import compute_all_metrics
from src.utils.paths import get_latest_model

def parse_args():
    parser = argparse.ArgumentParser(description="Calcular métricas de un modelo entrenado.")

    parser.add_argument("--alg", type=str, choices=["ppo", "a2c", "ql"], required=True)

    # Evaluación (común)
    parser.add_argument("--episodes", type=int, default=114, help="Número de episodios de evaluación.")
    parser.add_argument("--modo-eval", type=str, choices=["markov", "historico"], default="historico")

    # SB3 sliding window (PPO/A2C)
    parser.add_argument("--window-weeks", type=int, default=156)
    parser.add_argument("--stride-weeks", type=int, default=52)
    parser.add_argument("--n-envs", type=int, default=8)

    # Q-learning
    parser.add_argument("--num-pasos", type=int, default=155, help="Pasos por episodio (solo QL).")

    parser.add_argument("--save", action="store_true", help="Guardar métricas en un CSV.")

    return parser.parse_args()

def _latest_paths(alg: str):
    """
    get_latest_model puede devolver:
      - (model_path, vecnorm_path)  [PPO]
      - (model_path, None)          [A2C/QL]
      - model_path                  
    """
    out = get_latest_model(alg)
    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            return out[0], out[1]
        if len(out) == 1:
            return out[0], None
    return out, None

def main():
    args = parse_args()

    # ===============================
    # Elegir agente
    # ===============================
    if args.alg == "ppo":
        agent = PPOAgent()
    elif args.alg == "a2c":
        agent = A2CAgent()
    else:
        agent = QLearningAgent()

    # ===============================
    # Buscar el modelo más reciente
    # ===============================
    model_path, vecnorm_path = _latest_paths(args.alg)

    # ===============================
    # Cargar
    # ===============================
    if args.alg == "ppo":
        agent.load(model_path, vecnorm_path, modo_eval=args.modo_eval)
    elif args.alg == "a2c":
        agent.load(model_path, modo_eval=args.modo_eval)
    else:
        agent.load(model_path)

    # ===============================
    # Evaluar
    # ===============================
    print("\n=== Evaluando el modelo... ===\n")

    if args.alg in ("ppo", "a2c"):
        df_avg, df_all = agent.evaluate(
            n_eval_episodes=args.episodes,
            window_weeks=args.window_weeks,
            stride_weeks=args.stride_weeks,
            n_envs=args.n_envs,
            modo_eval=args.modo_eval,
        )
    else:
        df_avg, df_all = agent.evaluate(
            n_eval_episodes=args.episodes,
            num_pasos=args.num_pasos,
        )

    # ===============================
    # Métricas
    # ===============================
    print("\n=== Calculando métricas... ===\n")
    metrics = compute_all_metrics(df_avg, df_all)

    for k, v in metrics.items():
        print(f"{k:25s} : {v}")

    # ===============================
    # Guardar si corresponde
    # ===============================
    if args.save:
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"metrics_{args.alg}_{args.modo_eval}.csv"

        import pandas as pd
        df = pd.DataFrame([metrics])
        df.to_csv(out_path, index=False)
        print(f"\nMétricas guardadas en {out_path}\n")

if __name__ == "__main__":
    main()