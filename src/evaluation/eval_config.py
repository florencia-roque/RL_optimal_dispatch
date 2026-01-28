# src/evaluation/eval_config.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from src.utils.paths import timestamp, mode_tag

# =========================
# Config común (header)
# =========================

@dataclass(frozen=True)
class EvalHeader:
    fecha: str
    mode_tag_str: str
    deterministico: int
    mode_eval: str

def build_eval_header_from_env(*, env, mode_eval: str, multiple_seeds: bool) -> EvalHeader:
    """
    Para agentes que tienen env: arma fecha + mode_tag_str.
    """
    inner = env.unwrapped
    deterministico = int(getattr(inner, "DETERMINISTICO", 0))
    modo = str(mode_eval).lower()

    fecha = timestamp()
    mode_tag_str = mode_tag(deterministico, modo, multiple_seeds=multiple_seeds)

    return EvalHeader(
        fecha=fecha,
        mode_tag_str=mode_tag_str,
        deterministico=deterministico,
        mode_eval=modo,
    )

# =========================
# Config para SB3 paralelo
# =========================

@dataclass(frozen=True)
class EvalContext:
    fecha: str
    mode_tag_str: str
    deterministico: int
    mode_eval: str
    env_fns: list[Callable[[], object]]  # callables que construyen envs

def build_sb3_eval_context(
    *,
    alg: str,
    n_envs: int,
    mode_eval: str = "historico",
    deterministico: int = 0,
    seed = None,
    multiple_seeds: bool = False
) -> EvalContext:
    """
    Para PPO/A2C:
    - arma env_fns (n envs)
    - crea un env temporal para leer si es deterministico (y validar modo)
    - arma fecha + mode_tag_str
    """
    from src.environment.env_factory import make_eval_env

    modo = str(mode_eval).lower()

    env_fns = [lambda m=modo: make_eval_env(alg, modo=m, deterministico=deterministico, seed=seed + i if seed is not None else None) for i in range(n_envs)]

    tmp_env = env_fns[0]()
    inner = tmp_env.unwrapped
    deterministico = int(getattr(inner, "DETERMINISTICO", 0))
    tmp_env.close()

    fecha = timestamp()
    mode_tag_str = mode_tag(deterministico, modo, multiple_seeds=multiple_seeds)

    return EvalContext(
        fecha=fecha,
        mode_tag_str=mode_tag_str,
        deterministico=deterministico,
        mode_eval=modo,
        env_fns=env_fns,
    )