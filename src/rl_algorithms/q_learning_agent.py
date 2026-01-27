# src/rl_algorithms/qlearning_agent.py

from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.paths import (
    timestamp,
    mode_tag,
    training_paths,
)
from src.utils.io import (
    save_q_table,
    load_q_table,
)
from src.environment.env_factory import make_train_env, make_eval_env
from src.evaluation.eval_outputs import save_eval_outputs
from src.environment.utils_tabular import politica_optima
from src.evaluation.eval_config import build_eval_header_from_env
from src.utils.callbacks import LiveRewardPlotter

import optuna

class QLearningAgent:
    """
    Clase para entrenar y evaluar Q-Learning tabular en el entorno Hydro-Thermal Tabular.
    """
    def __init__(self, modo="ql", deterministico=0, seed=None):
        self.alg = modo
        self.env = None
        self.Q = None
        self.deterministico = deterministico
        self.seed = seed

    def train(self, total_episodes=3000, hparams=None, trial=None):
        print("Comienzo de entrenamiento Q-learning...")
        t0 = time.perf_counter()

        self.env = make_train_env("ql", deterministico=self.deterministico, seed=self.seed)
        inner = self.env.unwrapped

        # Hiperparámetros: usar sugerencias de Optuna o valores por defecto
        self.alpha = hparams.get("alpha", 0.001) if hparams else 0.001
        self.gamma = hparams.get("gamma", 0.99) if hparams else 0.99
        self.epsilon = hparams.get("epsilon", 0.01) if hparams else 0.01

        # hiperparametros hallados por optuna (hardcodeados!)
        self.alpha = 0.0069059394803614
        self.gamma = 0.9974407281619924
        self.epsilon = 0.0817594479135859

        # Inicializar Q en el agente
        n_states = inner.observation_space.n
        n_actions = inner.action_space.n
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

        modo_ent = inner.MODO
        fecha = timestamp()
        mode_tag_str = mode_tag(self.deterministico, modo_ent, multiple_seeds=False)

        paths = training_paths(self.alg, fecha, mode_tag_str)
        qtable_path = paths["model_path"]
        fig_path = paths["fig_path"]

        plotter = LiveRewardPlotter(window=100, refresh_every=10, filename=str(fig_path))

        comienzo_ultimos_100 = False
        reward_ultimos_100_episodios = 0.0

        rewards_window = [] # Para calcular el promedio móvil

        for episode in range(total_episodes):
            if episode % 100 == 0:
                print(f"Episodio: {episode}")

            if episode == total_episodes - 100:
                comienzo_ultimos_100 = True

            obs, _info = self.env.reset()
            idx = int(obs)

            done = False
            reward_episodio = 0.0

            while not done:
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(n_actions)
                else:
                    a = int(np.argmax(self.Q[idx]))

                next_obs, reward, terminated, truncated, _info = self.env.step(a)
                done = bool(terminated or truncated)
                next_idx = int(next_obs)

                self.Q[idx, a] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_idx]) - self.Q[idx, a]
                )

                reward_episodio += float(reward)
                idx = next_idx

            # --- LÓGICA DE OPTUNA (PRUNING) ---
            if trial is not None:
                rewards_window.append(reward_episodio)
                if len(rewards_window) > 100: rewards_window.pop(0)
                        
                # Reportar cada 50 episodios para no saturar
                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(rewards_window)
                    trial.report(avg_reward, episode)
                            
                    if trial.should_prune():
                        raise optuna.TrialPruned()    
            
            if comienzo_ultimos_100:
                reward_ultimos_100_episodios += float(reward_episodio)

            plotter.update(reward_episodio)

        plotter.close()
        print("La recompensa acumulada promedio de los ultimos 100 episodios en training fue:", reward_ultimos_100_episodios/100)
        save_q_table(self.Q, qtable_path)

        dt = (time.perf_counter() - t0) / 60
        print(f"Entrenamiento Q-learning completado en {dt:.2f} minutos")

    def load(self, qtable_path: Path, mode_eval="historico"):
        print(f"Cargando Q-table desde {qtable_path}...")
        self.Q = load_q_table(qtable_path)
        print("Q-table cargada.")

        self.env = make_eval_env("ql", modo=mode_eval, deterministico=self.deterministico, seed=self.seed)
        return self.env

    def evaluate(self, n_eval_episodes=114, num_pasos=None, mode_eval="historico", eval_seed=None, multiple_seeds=False):
        if self.env is None or self.Q is None:
            raise RuntimeError("Primero cargar o entrenar el agente Q-learning.")

        inner = self.env.unwrapped

        current_seed = eval_seed if eval_seed is not None else self.seed

        if current_seed is not None:
             inner.reset(seed=current_seed)

        if self.deterministico == 0 and hasattr(inner, "MODO"):
            if inner.MODO != mode_eval:
                print(f"[INFO] Cambiando modo del entorno de '{inner.MODO}' a '{mode_eval}' para evaluación.")
                inner.MODO = mode_eval

        if num_pasos is None:
            num_pasos = inner.T_MAX + 1

        if self.deterministico == 0:
            print("Evaluando con modo:", mode_eval)

        # Ajuste por histórico
        reset_con_start_week = (self.deterministico == 0 and mode_eval == "historico")
        if reset_con_start_week:
            max_start = len(inner.datos_historicos) - (inner.T_MAX + 1)
            max_eps = (max_start // 52) + 1
            if n_eval_episodes > max_eps:
                print(f"[WARN] n_eval_episodes={n_eval_episodes} > max={max_eps}. Ajustando.")
                n_eval_episodes = max_eps

        hdr = build_eval_header_from_env(env=self.env, mode_eval=mode_eval, multiple_seeds=multiple_seeds)

        print("Iniciando evaluación Q-learning...")

        politica = politica_optima(self.Q)
        resultados = []

        for ep in range(n_eval_episodes):
            reset_opts = {"start_week": 52 * ep} if reset_con_start_week else None

            obs, info = self.env.reset(options=reset_opts)
            idx = int(obs)

            for _ in range(num_pasos):
                action = int(politica[idx])

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)

                fila = dict(info)
                fila["episode_id"] = ep
                fila["action"] = action
                fila["reward"] = float(reward)
                resultados.append(fila)

                idx = int(next_obs)
                if done:
                    break

        df_all = pd.DataFrame(resultados)
        df_avg = df_all.groupby("tiempo", as_index=False).mean(numeric_only=True) if "tiempo" in df_all.columns else df_all

        reward_total, _, _ = save_eval_outputs(
            df_avg,
            df_all,
            alg=self.alg,
            fecha=hdr.fecha,
            mode_tag_str=hdr.mode_tag_str,
            estados_cols=[
                "volumen_discreto",
                "hidrologia",
                "tiempo",
                "aportes",
                "vertimiento",
                "volumen_turbinado",
            ],
            n_eval_episodes=n_eval_episodes,
        )

        print(f"Recompensa total en evaluación Q-learning: {reward_total:.2f}")
        return df_avg, df_all
    
    def evaluate_multiple_seed(
        self,
        n_eval_episodes=114,
        mode_eval="historico",
        seeds=None
    ): 
        
        resultados = {}
        if seeds is None: 
            raise ValueError("Se debe proporcionar una lista de semillas para la evaluación múltiple.")

        for seed in seeds:
            print(f"\nEvaluando con semilla: {seed}")
            df_avg, df_all = self.evaluate(
                n_eval_episodes=n_eval_episodes,
                mode_eval=mode_eval,
                eval_seed=seed,
                multiple_seeds=True
            )
            resultados[seed] = (df_avg, df_all)

        df_avg_mean = None
        df_all_mean = None
        for seed, (df_avg, df_all) in resultados.items():
            if df_avg_mean is None:
                df_avg_mean = df_avg.copy()
            else:
                df_avg_mean += df_avg

            if df_all_mean is None:
                df_all_mean = df_all.copy()
            else:
                df_all_mean += df_all 
                
        # Promediar los resultados
        n_seeds = len(seeds)
        df_avg_mean /= n_seeds
        df_all_mean /= n_seeds

        hdr = build_eval_header_from_env(env=self.env, mode_eval=mode_eval, multiple_seeds=True)

        reward_total, _, _ = save_eval_outputs(
            df_avg_mean,
            df_all_mean,
            alg=self.alg,
            fecha=hdr.fecha,
            mode_tag_str=hdr.mode_tag_str,
            estados_cols=[
                "volumen_discreto",
                "hidrologia",
                "tiempo",
                "aportes",
                "vertimiento",
                "volumen_turbinado",
            ],
            n_eval_episodes=n_eval_episodes,
        )
            
        print(f"\nRecompensa total promedio en evaluación múltiple Q-learning: {reward_total:.2f}")    
        return df_avg_mean, df_all_mean