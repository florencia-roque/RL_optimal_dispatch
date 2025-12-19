"""Wrappers del entorno.

Regla práctica:
- Todo lo que sea una transformación *del entorno Gym* (obs/action/reward) va acá.
- Todo lo que sea utilitario tabular (encode/decode bins, etc.) queda en
  src/environment/utils_tabular.py.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OneHotFlattenObs(gym.ObservationWrapper):
    """Convierte una observación dict en un vector numérico.

    Espera que el env subyacente exponga:
    - V_CLAIRE_MAX
    - N_HIDRO
    - T_MAX

    Output:
        [v_norm, one_hot(hidrologia), one_hot(tiempo)]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        inner = env.unwrapped
        self._vmax = float(getattr(inner, "V_CLAIRE_MAX"))
        self._n_hidro = int(getattr(inner, "N_HIDRO"))
        self._tmax = int(getattr(inner, "T_MAX"))

        dim = 1 + self._n_hidro + (self._tmax + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(dim,), dtype=np.float32
        )

    def observation(self, obs):
        v = float(obs["volumen"])
        v_norm = v / self._vmax

        h = int(obs["hidrologia"])
        hidro_oh = np.zeros(self._n_hidro, dtype=np.float32)
        hidro_oh[h] = 1.0

        t = int(obs["tiempo"])
        time_oh = np.zeros(self._tmax + 1, dtype=np.float32)
        time_oh[t] = 1.0

        return np.concatenate(([v_norm], hidro_oh, time_oh), axis=0)
