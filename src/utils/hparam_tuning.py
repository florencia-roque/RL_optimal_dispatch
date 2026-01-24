# src/utils/hparam_tuning.py

import optuna
import pandas as pd
from pathlib import Path 
from stable_baselines3.common.callbacks import BaseCallback   
from src.rl_algorithms import PPOAgent, A2CAgent, QLearningAgent
from src.utils.paths import timestamp

class TrialPruningCallback(BaseCallback):
    """
    Callback de Optuna para podar (prune) trials no prometedores.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str = "rollout/ep_rew_mean", verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.monitor = monitor

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # PPO actualiza las métricas al final del rollout
        # Accedemos al logger interno de SB3
        if self.logger and self.monitor in self.logger.name_to_value:
            value = self.logger.name_to_value[self.monitor]
            
            # Reportar a Optuna
            self.trial.report(value, self.num_timesteps)
            
            # Verificar si se debe podar
            if self.trial.should_prune():
                message = f"Trial {self.trial.number} pruned at step {self.num_timesteps} with value {value}"
                if self.verbose > 0:
                    print(message)
                raise optuna.TrialPruned(message)

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

        # Instanciar y Entrenar
        agent = self._create_agent()

        try:
            if self.alg == "ql":
                # Q-Learning manual: Pasamos el trial directo
                agent.train(total_episodes=1500, hparams=hparams, trial=trial)
            else:
                # PPO / A2C (SB3): Usamos el callback manual
                pruning_callback = TrialPruningCallback(trial, monitor="rollout/ep_rew_mean")
                agent.train(total_episodes=1000, hparams=hparams, extra_callback=pruning_callback)

            # Evaluación final (común para todos)
            df_avg, _ = agent.evaluate(n_eval_episodes=20, eval_seed=42)
            score = df_avg["reward"].mean()
            
            self._save_trial(trial.number, hparams, score)
            return score

        except optuna.TrialPruned:
            raise

    def _get_suggestions(self, trial):
        if self.alg == "ppo":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "gamma": trial.suggest_float("gamma", 0.98, 0.9999, log=False),
                "n_steps": trial.suggest_int("n_steps", 64, 256, log=True),
                "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
            }
        elif self.alg == "a2c":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
                "n_steps": trial.suggest_int("n_steps", 64, 256, log=True),
            }
        elif self.alg == "ql":
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                "gamma": trial.suggest_float("gamma", 0.8, 0.9999, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 0.1, log=True),
            }       

    def _create_agent(self):
        if self.alg == "ppo": return PPOAgent(deterministico=self.deterministico, seed=self.seed)
        if self.alg == "a2c": return A2CAgent(deterministico=self.deterministico, seed=self.seed)
        return QLearningAgent(deterministico=self.deterministico, seed=self.seed)

    def _save_trial(self, number, params, score):
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