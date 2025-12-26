# src/rl_algorithms/ppo_agent.py

from __future__ import annotations
import time
from pathlib import Path
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from src.utils.paths import (
    timestamp,
    mode_tag,
    training_paths,
)
from src.utils.io import (
    load_sb3_model,
    save_run_artifacts
)
from src.evaluation.evaluator_sb3 import evaluar_sb3_parallel_sliding
from src.environment.env_factory import make_train_env, make_eval_env
from src.evaluation.eval_outputs import save_eval_outputs
from src.evaluation.eval_config import build_sb3_eval_context
from src.utils.callbacks import LivePlotCallback

class PPOAgent:
    """
    Clase para entrenar y evaluar PPO en el entorno Hydro-Thermal Continuo.
    """
    def __init__(self, modo="ppo", n_envs=8):
        self.alg = modo
        self.n_envs = n_envs
        self.vec_env = None
        self.model = None

    def train(self, total_episodes=2000):
        print("Comienzo de entrenamiento PPO...")
        t0 = time.perf_counter()

        self.vec_env = DummyVecEnv([lambda: make_train_env("ppo") for _ in range(self.n_envs)])
        self.vec_env = VecMonitor(self.vec_env)
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        T_MAX = self.vec_env.get_attr("T_MAX")[0]
        total_timesteps = total_episodes * (T_MAX + 1)

        env0 = self.vec_env.envs[0].unwrapped
        deterministico = env0.DETERMINISTICO
        modo_ent = env0.MODO

        fecha = timestamp()
        mode_tag_str = mode_tag(deterministico, modo_ent)

        paths = training_paths(self.alg, fecha, mode_tag_str)
        fig_path = paths["fig_path"]
        model_path = paths["model_path"]
        vecnorm_path = paths["vecnorm_path"]

        policy_kwargs = dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            net_arch=dict(pi=[128], vf=[128]),
        )

        self.model = RecurrentPPO(
            MlpLstmPolicy,
            self.vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=104,
            gamma=0.99,
            ent_coef=0.005,
            learning_rate=3e-4,
            device="auto",
        )

        callback = LivePlotCallback(window=100, refresh_every=10, filename=str(fig_path))
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        save_run_artifacts(self.model, model_path, self.vec_env, vecnorm_path)

        dt = (time.perf_counter() - t0) / 60
        print(f"Entrenamiento completado en {dt:.2f} minutos")

    def load(self, model_path: Path, vecnorm_path: Path | None = None, modo_eval="historico"):
        print(f"Cargando modelo PPO desde {model_path}...")
        self.model = load_sb3_model(RecurrentPPO, model_path)
        print("Modelo cargado.")

        env_vec = DummyVecEnv([lambda: make_eval_env("ppo", modo=modo_eval)])

        if vecnorm_path is not None and vecnorm_path.exists():
            print(f"Cargando VecNormalize desde {vecnorm_path}...")
            env_vec = VecNormalize.load(str(vecnorm_path), env_vec)
            env_vec.training = False
            env_vec.norm_reward = False

        self.vec_env = env_vec
        return env_vec

    def evaluate(
        self,
        n_eval_episodes=114,
        window_weeks=156,
        stride_weeks=52,
        n_envs=8,
        modo_eval="historico",
    ):
        if self.model is None:
            raise RuntimeError("Primero cargar o entrenar el modelo PPO.")

        ctx = build_sb3_eval_context(alg=self.alg, n_envs=n_envs, modo_eval=modo_eval)

        print("Iniciando evaluación PPO...")

        df_avg, df_all = evaluar_sb3_parallel_sliding(
            self.model,
            env_fns=ctx.env_fns,
            n_eval_episodes=n_eval_episodes,
            window_weeks=window_weeks,
            stride_weeks=stride_weeks,
            deterministic=True,
        )

        reward_total, _, _ = save_eval_outputs(
            df_avg,
            df_all,
            alg=self.alg,
            fecha=ctx.fecha,
            mode_tag_str=ctx.mode_tag_str,
            estados_cols=["volumen", "hidrologia", "tiempo", "aportes", "vertimiento", "volumen_turbinado"],
            n_eval_episodes=n_eval_episodes,
        )

        print(f"Recompensa total en evaluación PPO: {reward_total:.2f}")
        return df_avg, df_all