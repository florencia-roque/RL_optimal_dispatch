# src/environment/utils_tabular.py
from __future__ import annotations

import numpy as np

def discretizar_volumen(env, v: float) -> int:
    """
    Discretiza un volumen continuo v en un bin [0, N_BINS_VOL-1]
    usando env.VOL_EDGES.
    """
    b = np.digitize([v], env.VOL_EDGES, right=False)[0] - 1
    return int(np.clip(b, 0, env.N_BINS_VOL - 1))
    
def codificar_estados(volumen_discreto,N_BINS_VOL,hidrologia,N_HIDRO,tiempo):
    # representar con un numero entre 0 y idx la tupla (v, h, t)
    idx = volumen_discreto + N_BINS_VOL * (hidrologia + N_HIDRO * tiempo)
    return idx
    
def codificar_estados(
    volumen_discreto: int,
    N_BINS_VOL: int,
    hidrologia: int,
    N_HIDRO: int,
    tiempo: int,
) -> int:
    """
    Codifica (v_bin, h, t) -> idx en [0, N_BINS_VOL*N_HIDRO*(T_MAX+1)-1]
    """
    idx = volumen_discreto + N_BINS_VOL * (hidrologia + N_HIDRO * tiempo)
    return int(idx)

def decodificar_estados(idx: int, N_BINS_VOL: int, N_HIDRO: int) -> tuple[int, int, int]:
    """
    Decodifica idx -> (v_bin, h, t)
    """
    assert idx >= 0
    volumen = idx % N_BINS_VOL
    hidrologia = (idx // N_BINS_VOL) % N_HIDRO
    tiempo = idx // (N_BINS_VOL * N_HIDRO)
    return int(volumen), int(hidrologia), int(tiempo)


def politica_optima(Q: np.ndarray) -> np.ndarray:
    """
    Devuelve la política greedy pi(s)=argmax_a Q[s,a]
    """
    return Q.argmax(axis=1)


def politica_cubo(env, policy: np.ndarray) -> np.ndarray:
    """
    Reordena policy (shape [N_STATES]) a cubo [t, h, v].
    """
    inner = env.unwrapped
    return policy.reshape(inner.T_MAX + 1, inner.N_HIDRO, inner.N_BINS_VOL)