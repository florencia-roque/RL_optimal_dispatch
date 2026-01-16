# main.py
"""
Punto de entrada centralizado para entrenar / evaluar algoritmos de RL
en el problema hidro-térmico.
"""

from __future__ import annotations
import argparse
import sys
from src.rl_algorithms import PPOAgent, A2CAgent, QLearningAgent
from src.utils.paths import get_latest_model
from src.utils.hparam_tuning import HyperparameterTuner

import random
import numpy as np
import torch

# Fijar semilla para reproducibilidad
seed = None

# # Estas líneas "inyectan" la seed en los motores de las librerías
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento / Evaluación de RL para despacho hidro-térmico."
    )

    parser.add_argument("--alg", choices=["ppo", "a2c", "ql"], required=True)
    parser.add_argument("--mode", choices=["train", "train_eval", "eval", "tune"], required=True)
    
    parser.add_argument("--det", type=int, choices=[0, 1], default=0, help="1 para usar aportes determinísticos, 0 para estocásticos")

    # Entrenamiento (nota: para QL el default lógico es 3000)
    parser.add_argument("--total-episodes", type=int, default=None)

    # Evaluación (común)
    parser.add_argument("--n-eval-episodes", type=int, default=114)

    # Sliding window (PPO / A2C)
    parser.add_argument("--window-weeks", type=int, default=156)
    parser.add_argument("--stride-weeks", type=int, default=52)
    parser.add_argument("--mode-eval", choices=["markov", "historico"], default="historico")

    # Q-learning
    parser.add_argument("--num-pasos", type=int, default=155)

    # Paralelismo (entrenamiento PPO/A2C y evaluación sliding)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--a2c-dummy", action="store_true")

    # Argumento opcional para el número de pruebas de Optuna
    parser.add_argument("--n-trials", type=int, default=50, help="Número de pruebas para Optuna")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print(f"Algoritmo: {args.alg}, Modo de ejecución: {args.mode}")

    # =========================
    # Modo Tuning con Optuna
    # =========================
    if args.mode == "tune":
        print(f"Iniciando Hyperparameter Tuning para {args.alg} con {args.n_trials} pruebas...")
        tuner = HyperparameterTuner(alg=args.alg, deterministico=args.det, seed=seed)
        tuner.tune(n_trials=args.n_trials)
        print(f"Tuning completado.")
        return
    
    # =========================
    # Instanciar agente
    # =========================
    if args.alg == "ppo":
        agent = PPOAgent(n_envs=args.n_envs, deterministico=args.det, seed=seed)

    elif args.alg == "a2c":
        agent = A2CAgent(
            n_envs=args.n_envs,
            use_subproc=not args.a2c_dummy,
            deterministico=args.det,
            seed=seed,
        )

    elif args.alg == "ql":
        agent = QLearningAgent(deterministico=args.det, seed=seed)

    else:
        sys.exit(f"Algoritmo no soportado: {args.alg}")

    # =========================
    # Entrenamiento
    # =========================
    if args.mode in ("train", "train_eval"):
        agent.train(total_episodes=args.total_episodes)

    # =========================
    # Evaluación
    # =========================
    if args.mode == "train_eval":
        if args.alg in ("ppo", "a2c"):
            agent.evaluate(
                n_eval_episodes=args.n_eval_episodes,
                window_weeks=args.window_weeks,
                stride_weeks=args.stride_weeks,
                n_envs=args.n_envs,
                mode_eval=args.mode_eval,
            )

        elif args.alg == "ql":
            agent.evaluate(
                n_eval_episodes=args.n_eval_episodes,
                num_pasos=args.num_pasos,
                mode_eval=args.mode_eval,
            )
    
    elif args.mode == "eval":
        model_path, vecnorm_path = get_latest_model(args.alg)
        if args.alg == "ppo":
            agent.load(model_path, vecnorm_path, mode_eval=args.mode_eval, n_envs=args.n_envs)
            agent.evaluate(
                n_eval_episodes=args.n_eval_episodes,
                window_weeks=args.window_weeks,
                stride_weeks=args.stride_weeks,
                mode_eval=args.mode_eval,
            )
            agent.close_env()
        elif args.alg == "a2c":
            agent.load(model_path, mode_eval=args.mode_eval)
            agent.evaluate(
                n_eval_episodes=args.n_eval_episodes,
                window_weeks=args.window_weeks,
                stride_weeks=args.stride_weeks,
                n_envs=args.n_envs,
                mode_eval=args.mode_eval,
            )
        else: # ql
            agent.load(model_path, mode_eval=args.mode_eval)
            agent.evaluate(
                n_eval_episodes=args.n_eval_episodes,
                num_pasos=args.num_pasos,
                mode_eval=args.mode_eval,
            )

if __name__ == "__main__":
    main()