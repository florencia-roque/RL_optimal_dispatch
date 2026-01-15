import optuna
from src.rl_algorithms import PPOAgent, A2CAgent, QLearningAgent

class HyperparameterTuner:
    def __init__(self, alg, deterministico=0, seed=42):
        self.alg = alg
        self.deterministico = deterministico
        self.seed = seed

    def objective(self, trial):
        # 1. Sugerir hiperparámetros
        if self.alg == "ppo":
            hparams = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "n_steps": trial.suggest_int("n_steps", 64, 256),
                "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
            }
            agent = PPOAgent(deterministico=self.deterministico, seed=self.seed)
        
        # 2. Entrenar con los parámetros sugeridos
        # Usamos pocos episodios para que el tuning no sea eterno
        agent.train(total_episodes=500, hparams=hparams)

        # 3. Evaluar
        # Optuna buscará MAXIMIZAR esta recompensa
        df_avg, _ = agent.evaluate(n_eval_episodes=20)
        return df_avg["reward"].mean()

    def tune(self, n_trials=50):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        
        print("Mejores parámetros:", study.best_params)
        return study.best_params