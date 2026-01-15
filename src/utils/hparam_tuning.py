# src/utils/hparam_tuning.py

import optuna
import pandas as pd
from pathlib import Path    
from src.rl_algorithms import PPOAgent, A2CAgent, QLearningAgent
from src.utils.paths import timestamp
# from optuna.integration import OptunaPruningCallback

class HyperparameterTuner:
    def __init__(self, alg, deterministico=0, seed=42):
        self.alg = alg
        self.deterministico = deterministico
        self.seed = seed
        self.log_file = Path(f"results/tuning/{self.alg}_tuning_{timestamp()}.csv")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def objective(self, trial):
        # Sugerir Hiperparámetros
        hparams = self._get_suggestions(trial)
        
        # # Configurar Agente
        # agent = self._setup_agent()

        # Crear el callback de poda para SB3
        # 'eval/mean_reward' es la métrica que SB3 reporta internamente
        # pruning_callback = OptunaPruningCallback(trial, monitor="eval/mean_reward")

        if self.alg == "ppo":
            agent = PPOAgent(deterministico=self.deterministico, seed=self.seed)

        # Pasamos los parámetros y el callback de poda
        try:
            agent.train(total_episodes=500, hparams=hparams, extra_callback=None)
            
            # Evaluación final para retornar el score a Optuna
            df_avg, _ = agent.evaluate(n_eval_episodes=114)
            score = df_avg["reward"].mean()
            
            self._log_trial(trial.number, hparams, score)
            return score

        except optuna.exceptions.TrialPruned:
            # Si Optuna decide podar el trial, lanzamos la excepción para que lo registre
            raise optuna.exceptions.TrialPruned()
    
    def _get_suggestions(self, trial):
        if self.alg == "ppo":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "n_steps": trial.suggest_int("n_steps", 64, 256),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
            }
        elif self.alg == "a2c":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "n_steps": trial.suggest_int("n_steps", 64, 256),
            }
        elif self.alg == "ql":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0),
                "discount_factor": trial.suggest_float("discount_factor", 0.8, 0.9999),
                "exploration_rate": trial.suggest_float("exploration_rate", 0.01, 1.0),
            }       

    def _log_trial(self, number, params, score):
        data = {"trial": number, "score": score, **params}
        df = pd.DataFrame([data])
        df.to_csv(self.log_file, mode='a', header=not self.log_file.exists(), index=False)

    def tune(self, n_trials=50):
        # Pruning (MedianPruner detiene trials por debajo de la mediana)
        study = optuna.create_study(
            direction="maximize", 
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(self.objective, n_trials=n_trials)
        print(f"Mejor score: {study.best_value} con params: {study.best_params}")