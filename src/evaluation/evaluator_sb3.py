# src/evaluation/evaluator_sb3.py

from __future__ import annotations
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

def evaluar_sb3_parallel_sliding(
    model,
    env_fns,
    n_eval_episodes: int,
    window_weeks: int = 156,
    stride_weeks: int = 52,
    deterministic: bool = True,
):
    vec_env = DummyVecEnv(env_fns)
    n_envs = vec_env.num_envs

    next_episode_id = 0
    active_episode_id = -np.ones(n_envs, dtype=int)

    state = None
    episode_start = np.ones((n_envs,), dtype=bool)

    resultados = []
    finished = 0
    steps_left = np.full(n_envs, window_weeks, dtype=int)

    # -----------------------
    # Reset inicial por env
    # -----------------------
    obs_list = [None] * n_envs

    for i in range(n_envs):
        if next_episode_id < n_eval_episodes:
            start_week = next_episode_id * stride_weeks
            obs_i, _info = vec_env.env_method("reset", indices=i, options={"start_week": start_week})[0]
            active_episode_id[i] = next_episode_id
            next_episode_id += 1
            steps_left[i] = window_weeks
        else:
            obs_i, _info = vec_env.env_method("reset", indices=i)[0]
            active_episode_id[i] = -1
            steps_left[i] = window_weeks

        obs_list[i] = obs_i

    obs = _stack_obs(obs_list)

    # =======================
    # Loop de evaluación
    # =======================
    while finished < n_eval_episodes:
        # Predict (recurrente o no)
        try:
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=deterministic
            )
        except TypeError:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs2, rewards, dones, infos = vec_env.step(action)

        rewards_np = np.asarray(rewards).reshape(-1)
        dones_np = np.asarray(dones).reshape(-1)
        acts = np.asarray(action)

        for i in range(n_envs):
            ep_id = int(active_episode_id[i])
            if ep_id < 0:
                continue

            info_i = infos[i] if isinstance(infos, (list, tuple)) else infos

            if acts.ndim == 1:
                act_i = float(acts[i])
            else:
                flat = acts[i].reshape(-1)
                act_i = float(flat[0]) if flat.size else np.nan

            fila = dict(info_i)
            fila["episode_id"] = ep_id
            fila["action"] = act_i
            fila["reward"] = float(rewards_np[i])
            resultados.append(fila)

        # Ventana corrediza: forzar done al agotar steps
        steps_left -= 1
        forced_done = steps_left <= 0
        real_done = dones_np.astype(bool)
        done_now = np.logical_or(real_done, forced_done)

        # Para LSTM: donde arranca episodio nuevo en el próximo paso
        episode_start = done_now.copy()

        # -----------------------
        # Reset selectivo por env
        # -----------------------
        obs_next = obs2  # base
        if isinstance(obs_next, dict):
            obs_next = {k: v.copy() for k, v in obs_next.items()}
        else:
            obs_next = obs_next.copy()

        for i in range(n_envs):
            if not done_now[i]:
                continue

            # Cerró un episodio activo
            if active_episode_id[i] >= 0:
                finished += 1

            if next_episode_id < n_eval_episodes:
                start_week = next_episode_id * stride_weeks
                obs_i, _info = vec_env.env_method("reset", indices=i, options={"start_week": start_week})[0]
                active_episode_id[i] = next_episode_id
                next_episode_id += 1
                steps_left[i] = window_weeks
            else:
                # Lo dejo inactivo
                obs_i, _info = vec_env.env_method("reset", indices=i)[0]
                active_episode_id[i] = -1
                steps_left[i] = window_weeks

            # Reemplazar obs del env reseteado
            if isinstance(obs_next, dict):
                for k in obs_next.keys():
                    obs_next[k][i] = obs_i[k]
            else:
                obs_next[i] = obs_i

        obs = obs_next

    df_all = pd.DataFrame(resultados)
    if "tiempo" in df_all.columns:
        df_avg = df_all.groupby("tiempo", as_index=False).mean(numeric_only=True)
    else:
        df_avg = df_all.copy()

    vec_env.close()
    return df_avg, df_all

def _stack_obs(obs_list):
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: np.stack([o[k] for o in obs_list], axis=0) for k in first.keys()}
    return np.stack(obs_list, axis=0)