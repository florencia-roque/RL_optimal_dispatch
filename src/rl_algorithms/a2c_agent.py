# src/rl_algorithms/a2c_agent.py

from __future__ import annotations
import time
from pathlib import Path
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from src.utils.paths import (
    timestamp,
    mode_tag,
    training_paths,
)
from src.utils.io import (
    load_sb3_model,
    save_run_artifacts
)
from src.environment.env_factory import make_train_env, make_eval_env
from src.evaluation.evaluator_sb3 import evaluar_sb3_parallel_sliding
from src.evaluation.eval_outputs import save_eval_outputs
from src.evaluation.eval_config import build_sb3_eval_context
from src.utils.callbacks import LivePlotCallback
from stable_baselines3.common.callbacks import CallbackList # Importar para combinar callbacks

class A2CAgent:
    """
    Clase para entrenar y evaluar A2C en el entorno Hydro-Thermal Continuo.
    """

    def __init__(self, modo="a2c", n_envs=8, use_subproc=True, deterministico=0, seed=None):
        """
        modo: string del algoritmo (siempre 'a2c')
        n_envs: número de entornos paralelos
        use_subproc: True para SubprocVecEnv, False para DummyVecEnv
        """
        self.alg = modo
        self.n_envs = n_envs
        self.use_subproc = use_subproc
        self.vec_env = None
        self.model = None
        self.deterministico = deterministico
        self.seed = seed

    # ============================================================
    # ENTRENAMIENTO
    # ============================================================

    def train(self, total_episodes=2000, hparams=None, extra_callback=None):
        """
        Entrena el agente.
        total_episodes: Número total de episodios para entrenar. 
        hparams: Diccionario de hiperparámetros sugeridos por Optuna.
        extra_callback: Callback de Optuna para pruning.
        """

        print("Comienzo de entrenamiento A2C...")
        t0 = time.perf_counter()

        # Crear entornos paralelos
        # SubprocVecEnv/DummyVecEnv esperan una lista de *callables* que construyen envs
        env_fns = [lambda: make_train_env("a2c", deterministico=self.deterministico, seed=self.seed) for _ in range(self.n_envs)]
        if self.use_subproc:
            vec_constructor = SubprocVecEnv(env_fns)
        else:
            vec_constructor = DummyVecEnv(env_fns)

        self.vec_env = VecMonitor(vec_constructor)

        # VecEnv expone get_attr también para SubprocVecEnv/DummyVecEnv
        T_MAX = self.vec_env.get_attr("T_MAX")[0]
        total_timesteps = total_episodes * (T_MAX + 1)

        env0 = self.vec_env.get_attr("unwrapped", 0)[0]
        modo_ent = env0.MODO

        fecha = timestamp()
        mode_tag_str = mode_tag(self.deterministico, modo_ent)

        # Rutas de guardado
        paths = training_paths(self.alg, fecha, mode_tag_str)
        fig_path = paths["fig_path"]
        model_path = paths["model_path"]

        learning_rate = hparams.get("learning_rate", 5e-5) if hparams else 5e-5
        gamma = hparams.get("gamma", 0.99) if hparams else 0.99
        n_steps = hparams.get("n_steps", 104) if hparams else 104 

        # Modelo A2C
        self.model = A2C(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            n_steps=n_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            device="auto",
            seed=self.seed,
        )

        # Callback para graficar recompensas en vivo   
        plot_callback = LivePlotCallback(
            window=100, refresh_every=10, filename=str(fig_path)
        )

        # Combinar con el callback de Optuna si se proporciona
        callbacks = [plot_callback]
        if extra_callback is not None:
            callbacks.append(extra_callback)
        
        callback_list = CallbackList(callbacks)

        # Entrenar usando la lista de callbacks
        self.model.learn(total_timesteps=total_timesteps, callback=callback_list)

        # Guardar modelo
        save_run_artifacts(self.model, model_path)

        dt = (time.perf_counter() - t0) / 60
        print(f"Entrenamiento A2C completado en {dt:.2f} minutos")

    # ============================================================
    # CARGA
    # ============================================================

    def load(self, model_path: Path, mode_eval="historico"):
        """
        Carga un modelo A2C entrenado.
        """
        print(f"Cargando modelo A2C desde {model_path}...")
        self.model = load_sb3_model(A2C, model_path)
        print("Modelo cargado.")

        # Crear env dummy para evaluación
        env_vec = DummyVecEnv([lambda: make_eval_env("a2c", modo=mode_eval, deterministico=self.deterministico, seed=self.seed)])

        self.vec_env = env_vec
        return env_vec

    # ============================================================
    # EVALUACIÓN
    # ============================================================

    def evaluate(
        self,
        n_eval_episodes=114,
        window_weeks=156,
        stride_weeks=52,
        n_envs=8,
        mode_eval="historico",
    ):
        if self.model is None:
            raise RuntimeError("Primero cargar o entrenar el modelo A2C.")

        if not hasattr(self, "ctx") or self.ctx is None:
            self.ctx = build_sb3_eval_context(
                alg=self.alg, 
                n_envs=n_envs, 
                mode_eval=mode_eval, 
                seed=self.seed
            )

        print("Iniciando evaluación A2C...")
        if self.deterministico == 0:
            print("Evaluando con modo:", mode_eval)

        df_avg, df_all = evaluar_sb3_parallel_sliding(
            self.model,
            env_fns=self.ctx.env_fns,
            n_eval_episodes=n_eval_episodes,
            window_weeks=window_weeks,
            stride_weeks=stride_weeks,
            deterministic=True,
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

        print(f"Recompensa total en evaluación A2C: {reward_total:.2f}")
        return df_avg, df_all