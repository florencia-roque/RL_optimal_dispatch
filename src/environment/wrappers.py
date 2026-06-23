# src/environment/wrappers.py

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

class OneHotFlattenObs(gym.ObservationWrapper):
    """Convierte una observación dict en un vector numérico.

    Espera que el env subyacente tenga las siguientes claves en la observación dict:
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

class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    """
    
    def __init__(self, env, max_steps=156, test_mode=False): 
        # Actualizado a gymnasium.spaces.Box
        assert isinstance(env.observation_space, gym.spaces.Box)
        
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0.0])), np.concatenate((high, [1.0]))
        
        # Actualizado a gymnasium.spaces.Box
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super().__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
            
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs):
        self._current_step = 0
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs), info

    def step(self, action):
        self._current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, obs):
        # El tiempo restante va de 1.0 (inicio) a 0.0 (fin)
        time_feature = 1.0 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        return np.concatenate((obs, [time_feature]))