# src/rl_algorithms/ppo_agent.py

from __future__ import annotations
import time
from pathlib import Path
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure
import torch

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
from src.environment.env_factory import make_train_env
from src.evaluation.eval_outputs import save_eval_outputs
from src.evaluation.eval_config import build_sb3_eval_context
from src.utils.callbacks import LivePlotCallback
from stable_baselines3.common.callbacks import CallbackList # Importar para combinar callbacks

class PPOAgent:
    """
    Clase para entrenar y evaluar PPO en el entorno Hydro-Thermal Continuo.
    """
    def __init__(self, modo="ppo", n_envs=16, deterministico=0, seed=None):
        print(f"Inicializando agente PPO con modo={modo}, n_envs={n_envs}, deterministico={deterministico}, seed={seed}")
        self.alg = modo
        self.n_envs = n_envs
        self.vec_env = None
        self.model = None
        self.deterministico = deterministico
        self.seed = seed

    def train(self, total_episodes=2000, hparams=None, extra_callback=None):
        """
        Entrena el agente.
        total_episodes: Número total de episodios para entrenar. 
        hparams: Diccionario de hiperparámetros sugeridos por Optuna.
        extra_callback: Callback de Optuna para pruning.
        """

        print("Comienzo de entrenamiento PPO...")
        t0 = time.perf_counter()

        self.vec_env = DummyVecEnv([lambda: make_train_env("ppo", deterministico=self.deterministico, seed=self.seed) for _ in range(self.n_envs)])
        self.vec_env = VecMonitor(self.vec_env)
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        T_MAX = self.vec_env.get_attr("T_MAX")[0]
        total_timesteps = total_episodes * (T_MAX + 1)

        env0 = self.vec_env.envs[0].unwrapped
        modo_ent = env0.MODO

        fecha = timestamp()
        mode_tag_str = mode_tag(self.deterministico, modo_ent)

        paths = training_paths(self.alg, fecha, mode_tag_str)
        fig_path = paths["fig_path"]
        model_path = paths["model_path"]
        vecnorm_path = paths["vecnorm_path"]

        policy_kwargs = dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            net_arch=dict(pi=[128], vf=[128]),
        )

        learning_rate = hparams.get("learning_rate", 5e-5) if hparams else 5e-5
        gamma = hparams.get("gamma", 0.99) if hparams else 0.99
        n_steps = hparams.get("n_steps", 104) if hparams else 104
        ent_coef = hparams.get("ent_coef", 0.005) if hparams else 0.005
        
        # Mejor score: -0.026321698005551536 con params: 
        # {'learning_rate': 5.908826045446591e-06, 'gamma': 0.9991879834579956, 'n_steps': 64, 'ent_coef': 0.00013368839981133497}
        learning_rate = 5.908826045446591e-06
        gamma = 0.9991879834579956
        n_steps = 64
        ent_coef = 0.00013368839981133497
        
        print(f"Hiperparámetros de entrenamiento PPO: learning_rate={learning_rate}, gamma={gamma}, n_steps={n_steps}, ent_coef={ent_coef}")

        self.model = RecurrentPPO(
            MlpLstmPolicy,
            self.vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=self.seed,
        )

        # Logger
        log_path = "./results/logs/ppo_training/"
        new_logger = configure(log_path, ["stdout", "csv", "log", "tensorboard"])
        self.model.set_logger(new_logger)
        
        # Callback para graficar recompensas en vivo   
        plot_callback = LivePlotCallback(window=100, refresh_every=10, filename=str(fig_path))

        # Combinar con el callback de Optuna si se proporciona
        callbacks = [plot_callback]
        if extra_callback is not None:
            callbacks.append(extra_callback)
        
        callback_list = CallbackList(callbacks)

        # Entrenar usando la lista de callbacks
        self.model.learn(total_timesteps=total_timesteps, callback=callback_list)

        save_run_artifacts(self.model, model_path, self.vec_env, vecnorm_path)

        dt = (time.perf_counter() - t0) / 60
        print(f"Entrenamiento completado en {dt:.2f} minutos")

    def load(self, model_path: Path, path_vec_normalize: Path | None = None, mode_eval="historico", n_envs=8):
        print(f"Cargando modelo PPO desde {model_path}...")
        self.model = load_sb3_model(RecurrentPPO, model_path)
        print("Modelo cargado.")

        # Construimos el entorno base correcto
        self.ctx = build_sb3_eval_context(alg=self.alg, n_envs=n_envs, mode_eval=mode_eval, seed=self.seed)
        
        base_env = DummyVecEnv(self.ctx.env_fns)

        # Aplicamos la normalización sobre ese entorno base
        if path_vec_normalize:
            print(f"Cargando estadísticas de normalización desde {path_vec_normalize}")
            self.vec_env = VecNormalize.load(path_vec_normalize, base_env)
            
            # Configuraciones críticas para evaluación
            self.vec_env.training = False 
            self.vec_env.norm_reward = False
        else:
            print("AVISO: No se proporcionó VecNormalize. Usando entorno base sin normalizar.")
            self.vec_env = base_env

        print("Agente cargado y entorno de evaluación listo.")

    def evaluate(
        self,
        n_eval_episodes=114,
        window_weeks=156,
        stride_weeks=52,
        n_envs = 8,
        mode_eval="historico",
        eval_seed=None
    ):
        if self.model is None:
            raise RuntimeError("Primero cargar o entrenar el modelo PPO.")

        if not hasattr(self, "ctx") or self.ctx is None:
            self.ctx = build_sb3_eval_context(
                alg=self.alg, 
                n_envs=n_envs, 
                mode_eval=mode_eval, 
                seed=eval_seed
            )

        print("Iniciando evaluación PPO...")
        if self.deterministico == 0:
            print("Evaluando con modo:", mode_eval)

        df_avg, df_all = evaluar_sb3_parallel_sliding(
            self.model,
            env_fns=self.ctx.env_fns,
            n_eval_episodes=n_eval_episodes,
            window_weeks=window_weeks,
            stride_weeks=stride_weeks,
            deterministic=True,
            vec_env=self.vec_env
        )

        reward_total, _, _ = save_eval_outputs(
            df_avg,
            df_all,
            alg=self.alg,
            fecha=self.ctx.fecha,
            mode_tag_str=self.ctx.mode_tag_str,
            estados_cols=["volumen", "hidrologia", "tiempo", "aportes", "vertimiento", "volumen_turbinado"],
            n_eval_episodes=n_eval_episodes,
        )

        print(f"Recompensa total en evaluación PPO: {reward_total:.2f}")
        return df_avg, df_all
    
    def close_env(self):
        if self.vec_env is not None:
            self.vec_env.close()
            print("Entorno de evaluación (VecNormalize) cerrado correctamente.")
            self.vec_env = None