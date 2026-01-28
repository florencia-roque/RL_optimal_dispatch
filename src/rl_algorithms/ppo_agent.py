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
from src.utils.paths import get_latest_model
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
        mode_tag_str = mode_tag(self.deterministico, modo_ent,multiple_seeds=False)

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

        # hiperparametros hallados por optuna (hardcodeados!)
        learning_rate = 1.9694437290033328e-05
        gamma = 0.9922058818530016
        n_steps = 137
        ent_coef = 0.0002918704130075

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

    def load(self, model_path: Path, path_vec_normalize: Path | None = None, mode_eval="historico", n_envs=8, eval_seed=None, multiple_seeds=False):
        print(f"Cargando modelo PPO desde {model_path.name}")
        self.model = load_sb3_model(RecurrentPPO, model_path)
        print("Modelo cargado.")

        # Construimos el entorno base correcto
        self.ctx = build_sb3_eval_context(alg=self.alg, n_envs=n_envs, mode_eval=mode_eval, deterministico=self.deterministico, seed=eval_seed, multiple_seeds=multiple_seeds)
        
        base_env = DummyVecEnv(self.ctx.env_fns)

        # Aplicamos la normalización sobre ese entorno base
        if path_vec_normalize:
            print(f"Cargando estadísticas de normalización desde {path_vec_normalize.name}")
            self.vec_env = VecNormalize.load(path_vec_normalize, base_env)
            
            # Configuraciones críticas para evaluación
            self.vec_env.training = False 
            self.vec_env.norm_reward = False
        else:
            raise ValueError("Se requiere path_vec_normalize para cargar VecNormalize en PPO.")

        print("Agente cargado y entorno de evaluación listo.")

    def evaluate(
        self,
        n_eval_episodes=114,
        window_weeks=156,
        stride_weeks=52,
        mode_eval="historico",
        eval_seed=None
    ):
        if self.model is None:
            raise RuntimeError("Primero cargar o entrenar el modelo PPO.")

        if not hasattr(self, "ctx") or self.ctx is None:
            raise RuntimeError("El contexto de evaluación no está construido. Llamar a 'load' primero.")

        # Si cambiamos de modo en la misma ejecución, regeneramos el contexto
        if self.ctx.mode_eval != mode_eval:
             self.ctx = build_sb3_eval_context(
                alg=self.alg, n_envs=self.n_envs, mode_eval=mode_eval, seed=eval_seed
            )

        print(f"[INFO] Configurando evaluación PPO (Modo: {mode_eval})...")

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

        print("Iniciando evaluación PPO...")
        if self.deterministico == 0:
            print("Evaluando con modo:", mode_eval)

        # # --- VERIFICACIÓN DE NORMALIZACIÓN ---
        # if isinstance(final_eval_env, VecNormalize):
        #     # Accedemos a la media de la primera variable (Volumen)
        #     mean_loaded = self.vec_env.obs_rms.mean[0]
        #     mean_eval = final_eval_env.obs_rms.mean[0]
            
        #     var_loaded = self.vec_env.obs_rms.var[0]
        #     var_eval = final_eval_env.obs_rms.var[0]
            
        #     print(f"\n[DEBUG] Verificando estadísticas de Normalización:")
        #     print(f"  > Media Volumen (Cargado): {mean_loaded:.4f} | (Usado en Eval): {mean_eval:.4f}")
        #     print(f"  > Varianza Volumen (Cargado): {var_loaded:.4f} | (Usado en Eval): {var_eval:.4f}")
            
        #     if abs(mean_loaded - mean_eval) < 1e-6:
        #         print("  > [OK] ¡Las estadísticas coinciden perfectamente!")
        #     else:
        #         print("  > [PELIGRO] Las estadísticas NO coinciden.")
        #     print("-" * 40 + "\n")
        # # -------------------------------------

        df_avg, df_all = evaluar_sb3_parallel_sliding(
            self.model,
            n_eval_episodes=n_eval_episodes,
            window_weeks=window_weeks,
            stride_weeks=stride_weeks,
            deterministic=True,
            vec_env=final_eval_env
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

        print(f"Recompensa total en evaluación PPO: {reward_total:.2f}")
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
        
        reward_total, _, _ = save_eval_outputs(
            df_avg_mean,
            df_all_mean,
            alg=self.alg,
            fecha=self.ctx.fecha,
            mode_tag_str=self.ctx.mode_tag_str,
            estados_cols=["volumen", "hidrologia", "tiempo", "aportes", "vertimiento", "volumen_turbinado"],
            n_eval_episodes=n_eval_episodes,
        )
            
        print(f"\nRecompensa total promedio en evaluación múltiple PPO: {reward_total:.2f}")    
        return df_avg_mean, df_all_mean
    
    def close_env(self):
        if self.vec_env is not None:
            self.vec_env.close()
            print("Entorno de evaluación (VecNormalize) cerrado correctamente.")
            self.vec_env = None