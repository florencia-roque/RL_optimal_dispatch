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

class QLearningAgent:
    """
    Clase para entrenar y evaluar Q-Learning tabular en el entorno Hydro-Thermal Tabular.
    """
    def __init__(self, modo="ql"):
        self.alg = modo
        self.env = None
        self.Q = None
        self.MODO = "historico"

    def train(self, total_episodes=3000):
        print("Comienzo de entrenamiento Q-learning...")
        t0 = time.perf_counter()

        self.env = make_train_env("ql")
        inner = self.env.unwrapped

        # Inicializar Q en el agente
        n_states = inner.observation_space.n
        n_actions = inner.action_space.n
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

        self.alpha = 0.001 # learning rate
        self.gamma = 0.99 # discount
        self.epsilon = 0.01 # exploración

        deterministico = inner.DETERMINISTICO
        modo_ent = inner.MODO
        fecha = timestamp()
        mode_tag_str = mode_tag(deterministico, modo_ent)

        paths = training_paths(self.alg, fecha, mode_tag_str)
        qtable_path = paths["model_path"]
        fig_path = paths["fig_path"]

        plotter = LiveRewardPlotter(window=100, refresh_every=10, filename=str(fig_path))

        for episode in range(total_episodes):
            if episode % 100 == 0:
                print(f"Episodio: {episode}")

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

            plotter.update(reward_episodio)

        plotter.close()
        save_q_table(self.Q, qtable_path)

        dt = (time.perf_counter() - t0) / 60
        print(f"Entrenamiento Q-learning completado en {dt:.2f} minutos")

    def load(self, qtable_path: Path, modo_eval="historico"):
        print(f"Cargando Q-table desde {qtable_path}...")
        self.Q = load_q_table(qtable_path)
        print("Q-table cargada.")

        self.MODO = modo_eval

        self.env = make_eval_env("ql", modo=modo_eval)
        return self.env

    def evaluate(self, n_eval_episodes=114, num_pasos=None):
        if self.env is None or self.Q is None:
            raise RuntimeError("Primero cargar o entrenar el agente Q-learning.")

        inner = self.env.unwrapped
        if num_pasos is None:
            num_pasos = inner.T_MAX + 1

        # Ajuste por histórico
        reset_con_start_week = (inner.DETERMINISTICO == 0 and inner.MODO == "historico")
        if reset_con_start_week:
            max_start = len(inner.datos_historicos) - (inner.T_MAX + 1)
            max_eps = max_start // 52
            if n_eval_episodes > max_eps:
                print(f"[WARN] n_eval_episodes={n_eval_episodes} > max={max_eps}. Ajustando.")
                n_eval_episodes = max_eps

        hdr = build_eval_header_from_env(env=self.env, modo_eval=self.MODO)

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