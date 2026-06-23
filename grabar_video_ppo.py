import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecNormalize
from src.evaluation.eval_config import build_sb3_eval_context

modelo_path = "results/models/ppo/RecurrentPPO_hydro_thermal_claire_continuous_2026-06-23_09-42-14_est_markov.zip"
vec_norm_path = "results/models/ppo/vecnorm_2026-06-23_09-42-14_est_markov.pkl"
video_folder = "results/videos/"
os.makedirs(video_folder, exist_ok=True)

det = 0 if "est" in modelo_path else 1

print("Configurando entorno con wrappers...")

ctx = build_sb3_eval_context(
    alg="ppo", 
    n_envs=1, 
    mode_eval="historico", 
    deterministico=det,
    seed=7123,
    multiple_seeds=False
)
eval_env = DummyVecEnv(ctx.env_fns)

eval_env.render_mode = "rgb_array"                   # Capa DummyVecEnv
eval_env.envs[0].unwrapped.render_mode = "rgb_array" # Capa Entorno Base (Núcleo)

eval_env.envs[0].unwrapped.metadata = {"render_fps": 1}

print("Cargando VecNormalize...")
eval_env = VecNormalize.load(vec_norm_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False
eval_env.render_mode = "rgb_array"

print("Iniciando grabador de video...")
eval_env = VecVideoRecorder(
    eval_env,
    video_folder=video_folder,
    record_video_trigger=lambda step: step == 0,
    video_length=156,
    name_prefix="ppo_hydro_dispatch_nuevo"
)

print("Cargando modelo RecurrentPPO...")
model = RecurrentPPO.load(modelo_path)

print("Grabando simulación (156 semanas)...")
obs = eval_env.reset()

lstm_states = None
episode_starts = [True]

for i in range(156):
    action, lstm_states = model.predict(
        obs,
        state=lstm_states,
        episode_start=episode_starts,
        deterministic=True
    )
    obs, _, dones, _ = eval_env.step(action)
    episode_starts = dones

# Cerrar y guardar el MP4
eval_env.close()
print(f"¡Éxito! Video guardado en: {video_folder}")