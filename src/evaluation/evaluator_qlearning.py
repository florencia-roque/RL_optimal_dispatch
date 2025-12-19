# src/evaluation/evaluator_qlearning.py
from __future__ import annotations

from typing import Callable, Tuple, Dict, Any, List
import numpy as np
import pandas as pd


def evaluar_qlearning_parallel_sliding(
    env_fn: Callable[[], Any],          # callable que crea un env NUEVO
    Q: np.ndarray,                      # Q-table [n_states, n_actions]
    n_eval_episodes: int,
    window_weeks: int = 156,            # 3 años
    stride_weeks: int = 52,             # ventana corrediza 1 año
    n_envs: int = 8,                    # paralelismo manual
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evalúa una Q-table con ventana corrediza en paralelo manual (multiples envs).

    - episodio k arranca en start_week = k * stride_weeks
    - cada episodio dura window_weeks pasos (forzado, aunque el env no termine)
    - política greedy: argmax(Q[s])
    """

    envs = [env_fn() for _ in range(n_envs)]
    try:
        resultados: List[Dict[str, Any]] = []

        next_episode_id = 0
        active_episode_id = -np.ones(n_envs, dtype=int)
        steps_left = np.zeros(n_envs, dtype=int)
        obs_idx = np.zeros(n_envs, dtype=int)

        # -------------------------
        # Reset inicial por env
        # -------------------------
        for i in range(n_envs):
            if next_episode_id < n_eval_episodes:
                start_week = next_episode_id * stride_weeks
                obs, info = envs[i].reset(options={"start_week": start_week})
                active_episode_id[i] = next_episode_id
                next_episode_id += 1
                steps_left[i] = window_weeks
                obs_idx[i] = int(obs)
            else:
                # env queda inactivo
                obs, info = envs[i].reset()
                active_episode_id[i] = -1
                steps_left[i] = 0
                obs_idx[i] = int(obs)

        finished = 0

        # =========================
        # Loop principal
        # =========================
        while finished < n_eval_episodes:
            for i in range(n_envs):
                ep_id = int(active_episode_id[i])
                if ep_id < 0:
                    continue

                if steps_left[i] <= 0:
                    # forzar fin de episodio por ventana
                    finished += 1
                    _reset_env_slot(envs, i, active_episode_id, steps_left, obs_idx,
                                    next_episode_id, n_eval_episodes, stride_weeks, window_weeks)
                    if active_episode_id[i] >= 0:
                        next_episode_id += 1
                    continue

                s = int(obs_idx[i])
                a = int(np.argmax(Q[s]))

                next_obs, reward, terminated, truncated, info = envs[i].step(a)
                done = bool(terminated or truncated)

                fila = dict(info) if isinstance(info, dict) else {}
                fila["episode_id"] = ep_id
                fila["action"] = a
                fila["reward"] = float(reward)
                resultados.append(fila)

                obs_idx[i] = int(next_obs)
                steps_left[i] -= 1

                # Si el env terminó antes, cerramos episodio igualmente
                if done:
                    finished += 1
                    _reset_env_slot(envs, i, active_episode_id, steps_left, obs_idx,
                                    next_episode_id, n_eval_episodes, stride_weeks, window_weeks)
                    if active_episode_id[i] >= 0:
                        next_episode_id += 1

        df_all = pd.DataFrame(resultados)

        if "tiempo" in df_all.columns:
            df_avg = df_all.groupby("tiempo", as_index=False).mean(numeric_only=True)
        else:
            df_avg = df_all.copy()

        return df_avg, df_all

    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass


def _reset_env_slot(
    envs: List[Any],
    i: int,
    active_episode_id: np.ndarray,
    steps_left: np.ndarray,
    obs_idx: np.ndarray,
    next_episode_id: int,
    n_eval_episodes: int,
    stride_weeks: int,
    window_weeks: int,
) -> None:
    """
    Resetea un slot i y le asigna un nuevo episodio si quedan.
    """
    if next_episode_id < n_eval_episodes:
        start_week = next_episode_id * stride_weeks
        obs, _info = envs[i].reset(options={"start_week": start_week})
        active_episode_id[i] = next_episode_id
        steps_left[i] = window_weeks
        obs_idx[i] = int(obs)
    else:
        # queda inactivo
        obs, _info = envs[i].reset()
        active_episode_id[i] = -1
        steps_left[i] = 0
        obs_idx[i] = int(obs)
