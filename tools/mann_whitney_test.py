# tools/mann_whitney_test.py

import sys
import os
# Agregar la carpeta raíz (RL_optimal_dispatch) al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from scipy.stats import mannwhitneyu
from src.utils.config import EVALUATIONS

# ubicaciones de escenarios para ppo y q-learning
eval_dir_ql = EVALUATIONS / "ql" / "eval_2026-07-13_09-22-40_est_historico"
eval_dir_ppo = EVALUATIONS / "ppo" / "eval_2026-07-13_08-48-54_est_historico"

rewards_por_escenario_ppo = []
rewards_por_escenario_ql = []

for i in range(114):
    # cargar los archivos de escenarios (escenario_0.csv,...,escenario_113.csv) en dataframes
    df_ql = pd.read_csv(eval_dir_ql / f"escenario_{i}.csv")
    df_ppo = pd.read_csv(eval_dir_ppo / f"escenario_{i}.csv")

    # en cada df de escenario calcular la suma de la columna reward y guardarlo en una lista de rewards
    rewards_por_escenario_ql.append(df_ql["reward"].sum())
    rewards_por_escenario_ppo.append(df_ppo["reward"].sum())


# Verificación de que ambos vectores tengan largo 114
assert len(rewards_por_escenario_ppo) == 114
assert len(rewards_por_escenario_ql) == 114

# Calculamos el test de Mann-Whitney U
# 'two-sided' evalúa si son diferentes, pero la estadística U nos da las victorias
res = mannwhitneyu(rewards_por_escenario_ppo, rewards_por_escenario_ql, alternative='two-sided')

# Convertimos la estadística U a Probabilidad de Mejora
# n_ppo * n_ql es el número total de comparaciones de pares (114 * 114 = 12996)
n_ppo, n_ql = len(rewards_por_escenario_ppo), len(rewards_por_escenario_ql)
prob_improvement = res.statistic / (n_ppo * n_ql)

print(f"Probabilidad de que PPO mejore a Q-Learning: {prob_improvement:.4f}")


