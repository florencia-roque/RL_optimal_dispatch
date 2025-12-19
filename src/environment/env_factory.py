"""src/environment/env_factory.py

Factory de entornos para entrenamiento y evaluación.

Objetivos
---------
- Centralizar construcción de envs (continuo vs tabular) según algoritmo.
- Evitar duplicación de código en PPO/A2C/Q-learning.

Convenciones actuales del proyecto
---------------------------------
- Entrenamiento: siempre MODO='markov'.
- Evaluación:
    * si el env es determinístico (DETERMINISTICO==1): se evalúa en la misma tira
      (no corresponde pedir modo por consola).
    * si el env es estocástico (DETERMINISTICO==0): se puede evaluar en 'markov'
      o 'historico'.

Nota
----
No mezclamos “algoritmo” con “modo”: el algoritmo define el tipo de env
(continuo/tabular) y el wrapper de observación; el modo es una configuración del env.
"""

from __future__ import annotations

from typing import Optional

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


def _prompt_modo() -> str:
    modo_eval = input("Ingrese modo de evaluación M para Markov o H para histórico: ")
    modo_eval = (modo_eval or "").strip().lower()
    if modo_eval == "m":
        return "markov"
    if modo_eval == "h":
        return "historico"
    print("Opción inválida. Usando 'markov' por defecto.")
    return "markov"


def make_base_env(alg: str):
    """Crea el entorno base (tipo de env + wrappers + TimeLimit), sin tocar MODO."""
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
    """Entorno para ENTRENAMIENTO (por convención, MODO='markov')."""
    env = make_base_env(alg)
    inner = env.unwrapped
    if hasattr(inner, "MODO"):
        inner.MODO = "markov"
    return env


def make_eval_env(alg: str, modo: Optional[str] = None):
    """Entorno para EVALUACIÓN.

    Parameters
    ----------
    alg:
        'ppo' | 'a2c' | 'ql' (alias aceptados).

    modo:
        - None: decide automáticamente.
            * determinístico -> no pide nada, usa el modo que tenga el env.
            * estocástico    -> pide por consola (markov/historico).
        - 'markov' / 'historico': fuerza el modo (solo si el env tiene atributo MODO).

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
    
    # Caso estocástico: si no nos pasan modo, lo pedimos.
    elif modo is None and hasattr(inner, "MODO"):
        modo = _prompt_modo()

    # Si el env implementa MODO, lo seteamos.
    if hasattr(inner, "MODO"):
        inner.MODO = str(modo).lower()

    return env