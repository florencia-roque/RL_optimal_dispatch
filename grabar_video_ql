import os
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from src.environment.hydrothermal_env_tabular import HydroThermalEnvTab
from src.rl_algorithms.q_learning_agent import QLearningAgent

modelo_path = "results/models/ql/Q_table_2026-01-28_15-25-09_est_markov.npy"
video_folder = "results/videos/"
os.makedirs(video_folder, exist_ok=True)

print("Configurando entorno tabular...")
def make_env():
    env = HydroThermalEnvTab(seed=42)
    env.MODO = "historico"
    env.DETERMINISTICO = 0
    env.render_mode = "rgb_array"
    return env

eval_env = DummyVecEnv([make_env])

eval_env.envs[0].unwrapped.metadata = {"render_fps": 1}

print("Iniciando grabador de video...")
eval_env = VecVideoRecorder(
    eval_env,
    video_folder=video_folder,
    record_video_trigger=lambda step: step == 0,
    video_length=156,
    name_prefix="ql_hydro_dispatch"
)

print("Cargando agente Q-Learning...")
agent = QLearningAgent(deterministico=0, seed=42)
agent.load(modelo_path) # Cargamos la Q-table entrenada

print("Grabando simulación de Q-Learning (156 semanas)...")
obs = eval_env.reset()

for i in range(156):
    s = obs[0]
    
    idx = int(s)
    
    action = int(np.argmax(agent.Q[idx]))
        
    obs, _, dones, _ = eval_env.step([action])

# Cerrar y guardar el MP4
eval_env.close()
print(f"¡Éxito! Video de Q-Learning guardado en: {video_folder}")