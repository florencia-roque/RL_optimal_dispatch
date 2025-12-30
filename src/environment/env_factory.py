# src/environment/env_factory.py

from __future__ import annotations
from gymnasium.wrappers import TimeLimit
from src.environment.hydrothermal_env_continuous import HydroThermalEnvCont
from src.environment.hydrothermal_env_tabular import HydroThermalEnvTab
from src.environment.wrappers import OneHotFlattenObs

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_alg(alg: str) -> str:
    a = alg.strip().lower()
    if a in {"qlearning", "q_learning", "q-learning", "ql"}:
        return "ql"
    if a in {"ppo"}:
        return "ppo"
    if a in {"a2c"}:
        return "a2c"
    return a

def make_base_env(alg: str):
    """Crea el entorno base (tipo de env + wrappers + TimeLimit)"""
    alg_n = _normalize_alg(alg)

    if alg_n in {"ppo", "a2c"}:
        env = HydroThermalEnvCont()
        env = OneHotFlattenObs(env)
    elif alg_n == "ql":
        env = HydroThermalEnvTab()
    else:
        raise ValueError(f"Algoritmo desconocido en env_factory: {alg}")

    inner = env.unwrapped
    T_MAX = getattr(inner, "T_MAX", None)
    if T_MAX is None:
        raise AttributeError("El entorno no tiene atributo T_MAX definido.")

    return TimeLimit(env, max_episode_steps=int(T_MAX) + 1)

# -----------------------------------------------------------------------------
# Entrenamiento / Evaluación
# -----------------------------------------------------------------------------

def make_train_env(alg: str):
    """Entorno para ENTRENAMIENTO"""
    env = make_base_env(alg)
    inner = env.unwrapped
    if hasattr(inner, "MODO"):
        inner.MODO = "markov"
    return env

def make_eval_env(alg: str, modo: str):
    """Entorno para EVALUACIÓN.

    Parameters
    ----------
    alg:
        'ppo' | 'a2c' | 'ql' (alias aceptados).

    modo:
        - 'markov' / 'historico': fuerza el modo.

    Returns
    -------
    env: gymnasium.Env
        Environment ya envuelto con TimeLimit (+ wrappers si aplica).
    """
    env = make_base_env(alg)
    inner = env.unwrapped

    deterministico = int(getattr(inner, "DETERMINISTICO", 0))

    # Caso determinístico: no hay aleatoriedad; no corresponde elegir modo.
    if deterministico == 1:
        return env
    
    if hasattr(inner, "MODO"):
        inner.MODO = str(modo).lower()
    
    return env