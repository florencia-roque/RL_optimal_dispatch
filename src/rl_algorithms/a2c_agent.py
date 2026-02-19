# src/rl_algorithms/a2c_agent.py

from __future__ import annotations
import time
from pathlib import Path
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
import torch
from src.utils.paths import get_latest_model
import pandas as pd

from src.utils.paths import (
    timestamp,
    mode_tag,
    training_paths,
)
from src.utils.io import (
    load_sb3_model,
    save_run_artifacts
)
from src.environment.env_factory import make_train_env
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
            
        self.vec_env = DummyVecEnv([lambda: make_train_env("a2c", deterministico=self.deterministico, seed=self.seed) for _ in range(self.n_envs)])
        self.vec_env = VecMonitor(self.vec_env)
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        T_MAX = self.vec_env.get_attr("T_MAX")[0]
        total_timesteps = total_episodes * (T_MAX + 1)

        env0 = self.vec_env.envs[0].unwrapped
        modo_ent = env0.MODO

        fecha = timestamp()
        mode_tag_str = mode_tag(self.deterministico, modo_ent, multiple_seeds=False)

        paths = training_paths(self.alg, fecha, mode_tag_str)
        fig_path = paths["fig_path"]
        model_path = paths["model_path"]
        vecnorm_path = paths["vecnorm_path"]

        learning_rate = hparams.get("learning_rate", 5e-5) if hparams else 5e-5
        gamma = hparams.get("gamma", 0.99) if hparams else 0.99
        n_steps = hparams.get("n_steps", 104) if hparams else 104 

        print(f"Hiperparámetros de entrenamiento A2C: learning_rate={learning_rate}, gamma={gamma}, n_steps={n_steps}")
        
        self.model = A2C(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            n_steps=n_steps,
            learning_rate=learning_rate,
            gamma=gamma,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=self.seed,
        )

        # Callback para graficar recompensas en vivo   
        plot_callback = LivePlotCallback(window=100, refresh_every=10, filename=str(fig_path))

        # Combinar con el callback de Optuna si se proporciona
        callbacks = [plot_callback]
        if extra_callback is not None:
            callbacks.append(extra_callback)
        
        callback_list = CallbackList(callbacks)

        # Entrenar usando la lista de callbacks
        self.model.learn(total_timesteps=total_timesteps, callback=callback_list)

        # Guardar modelo
        save_run_artifacts(self.model, model_path, self.vec_env, vecnorm_path)

        dt = (time.perf_counter() - t0) / 60
        print(f"Entrenamiento A2C completado en {dt:.2f} minutos")

    # ============================================================
    # CARGA
    # ============================================================

    def load(self, model_path: Path, path_vec_normalize: Path | None = None, mode_eval="historico", n_envs=8, eval_seed=None, multiple_seeds=False):
        print(f"Cargando modelo A2C desde {model_path}...")
        self.model = load_sb3_model(A2C, model_path)
        print("Modelo cargado.")

        # Construimos el entorno base correcto 
        self.ctx = build_sb3_eval_context(alg=self.alg, n_envs=n_envs, mode_eval=mode_eval, seed=eval_seed, multiple_seeds=multiple_seeds)

        env_vec = DummyVecEnv(self.ctx.env_fns)

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
        mode_eval="historico",
        eval_seed=None
    ):
        
        if self.model is None:
            raise RuntimeError("Primero cargar o entrenar el modelo A2C.")

        if not hasattr(self, "ctx") or self.ctx is None:
            raise RuntimeError("El contexto de evaluación no está construido. Llamar a 'load' primero.")

        # Si cambiamos de modo en la misma ejecución, regeneramos el contexto
        if self.ctx.mode_eval != mode_eval:
             self.ctx = build_sb3_eval_context(
                alg=self.alg, n_envs=self.n_envs, mode_eval=mode_eval, seed=eval_seed
            )

        print(f"[INFO] Configurando evaluación A2C (Modo: {mode_eval})...")

        # Construir el entorno FÍSICO nuevo (Correcto: Histórico o Markov según pedido)
        # Usamos DummyVecEnv para evaluación determinística y secuencial
        eval_vec_env = DummyVecEnv(self.ctx.env_fns)

        # Aplicar NORMALIZACIÓN (Sincronizar con Training)
        # Si el modelo fue entrenado con VecNormalize, el entorno de evaluación 
        # debe normalizar las inputs usando las MISMAS estadísticas.
        if self.vec_env is not None and isinstance(self.vec_env, VecNormalize):
            # print("[DEBUG] Sincronizando estadísticas de normalización...")
            norm_eval_env = VecNormalize(eval_vec_env, training=False, norm_reward=False, clip_obs=10.0)
            
            # COPIAR LAS ESTADÍSTICAS DEL ENTRENAMIENTO
            norm_eval_env.obs_rms = self.vec_env.obs_rms
            norm_eval_env.ret_rms = self.vec_env.ret_rms
            
            # Usamos este entorno envuelto
            final_eval_env = norm_eval_env
        else:
            # Si no hubo normalización en train, usamos el env crudo
            final_eval_env = eval_vec_env

        print("Iniciando evaluación A2C...")
        if self.deterministico == 0:
            print("Evaluando con modo:", mode_eval)

        df_avg, df_all = evaluar_sb3_parallel_sliding(
            self.model,
            n_eval_episodes=n_eval_episodes,
            window_weeks=window_weeks,
            stride_weeks=stride_weeks,
            deterministic=True,
            vec_env=final_eval_env,
        )

        # Limpieza
        final_eval_env.close()

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
    
    def evaluate_multiple_seed(
        self,
        n_eval_episodes=114,
        window_weeks=156,
        stride_weeks=52,
        n_envs = 8,
        mode_eval="historico",
        seeds=None
    ): 
        resultados = {}
        if seeds is None: 
            raise ValueError("Se debe proporcionar una lista de semillas para la evaluación múltiple.")

        for seed in seeds:
            print(f"\nEvaluando con semilla: {seed}")
            model_path, vecnorm_path = get_latest_model(self.alg)
            self.load(model_path, vecnorm_path, mode_eval=mode_eval, n_envs=n_envs, eval_seed=seed, multiple_seeds=True)
            df_avg, df_all = self.evaluate(
                n_eval_episodes=n_eval_episodes,
                window_weeks=window_weeks,
                stride_weeks=stride_weeks,
                mode_eval=mode_eval
            )
            resultados[seed] = (df_avg, df_all)

        list_df_avg = [res[0] for res in resultados.values()]
        list_df_all = [res[1] for res in resultados.values()]
        
        def aggregate_dataframes(df_list):
            # Concatenamos todos los DFs uno sobre otro
            df_concat = pd.concat(df_list)
            
            # Agrupamos por el índice (asumiendo que el índice representa el paso de tiempo/episodio)
            grouped = df_concat.groupby(df_concat.index)
            
            cols_continuas = ["volumen", "volumen_turbinado", "energia_hidro", "energia_eolica",
                                   "energia_solar", "energia_biomasa", "energia_renovable", "energia_termico_bajo", "energia_termico_alto", "energia_exportada",
                                   "costo_termico", "ingreso_exportacion", "demanda", "demanda_residual", "fraccion_turbinado", "qt_max_fisico", "energia_hidro_max_frac",
                                   "energia_hidro_obj", "aportes", "vertimiento", "action", "reward"]
            cols_discretas = ["hidrologia", "tiempo", "episode_id"]

            # Diccionario de agregación
            reglas = {col: 'mean' for col in cols_continuas}
            for col in cols_discretas:
                # La moda puede devolver múltiples valores, tomamos el primero [0]
                reglas[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else None

            return grouped.agg(reglas)

        # Calcular los promedios/modas de cada columna para cada paso de tiempo/episodio
        df_avg_mean = aggregate_dataframes(list_df_avg)
        df_all_mean = aggregate_dataframes(list_df_all)

        # Guardar resultados
        reward_total, _, _ = save_eval_outputs(
            df_avg_mean,
            df_all_mean,
            alg=self.alg,
            fecha=self.ctx.fecha,
            mode_tag_str=f"{self.ctx.mode_tag_str}_promedio_seeds",
            estados_cols=["volumen", "hidrologia", "tiempo", "aportes", "vertimiento", "volumen_turbinado"],
            n_eval_episodes=n_eval_episodes,
        )
            
        print(f"\nRecompensa total promedio en evaluación múltiple A2C: {reward_total:.2f}")    
        return df_avg_mean, df_all_mean