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
    modo_eval: str


def build_eval_header_from_env(*, env, modo_eval: str | None = None) -> EvalHeader:
    """
    Para agentes que ya tienen env (ej: QL): arma fecha + mode_tag_str estándar.
    """
    inner = env.unwrapped
    deterministico = int(getattr(inner, "DETERMINISTICO", 0))
    modo = str(modo_eval or getattr(inner, "MODO", "markov")).lower()

    fecha = timestamp()
    mode_tag_str = mode_tag(deterministico, modo)

    return EvalHeader(
        fecha=fecha,
        mode_tag_str=mode_tag_str,
        deterministico=deterministico,
        modo_eval=modo,
    )


# =========================
# Config para SB3 paralelo
# =========================

@dataclass(frozen=True)
class EvalContext:
    fecha: str
    mode_tag_str: str
    deterministico: int
    modo_eval: str
    env_fns: list[Callable[[], object]]  # callables que construyen envs


def build_sb3_eval_context(
    *,
    alg: str,
    n_envs: int,
    modo_eval: str = "historico",
) -> EvalContext:
    """
    Para PPO/A2C:
    - arma env_fns (N envs)
    - crea un env temporal para leer determinismo (y validar modo)
    - arma fecha + mode_tag_str
    """
    from src.environment.env_factory import make_eval_env

    modo = str(modo_eval).lower()

    env_fns = [lambda m=modo: make_eval_env(alg, modo=m) for _ in range(n_envs)]

    tmp_env = env_fns[0]()
    inner = tmp_env.unwrapped
    deterministico = int(getattr(inner, "DETERMINISTICO", 0))
    tmp_env.close()

    fecha = timestamp()
    mode_tag_str = mode_tag(deterministico, modo)

    return EvalContext(
        fecha=fecha,
        mode_tag_str=mode_tag_str,
        deterministico=deterministico,
        modo_eval=modo,
        env_fns=env_fns,
    )
