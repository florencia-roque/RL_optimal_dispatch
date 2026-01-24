# src/utils/callbacks.py

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

# Determinar si el backend actual es interactivo
_INTERACTIVE_BACKENDS = {
    "Qt5Agg", "QtAgg", "TkAgg", "WXAgg", "GTK3Agg", "MacOSX"
}
_IS_INTERACTIVE_BACKEND = matplotlib.get_backend() in _INTERACTIVE_BACKENDS

# ============================================================
# Base común
# ============================================================

class _LivePlotBase:
    """
    Lógica común para graficar reward por episodio + media móvil, y guardar fig al final.
    """

    def _init_plot(
        self,
        window: int,
        refresh_every: int,
        title: Optional[str],
        filename: str,
        expected_total_episodes: Optional[int] = None,
    ) -> None:
        self.window = int(window)
        self.refresh_every = int(refresh_every)
        self.filename = str(filename)
        self.expected_total_episodes = expected_total_episodes

        self.rewards_ep: list[float] = []
        self.moving_avg: list[float] = []

        # Interactivo sólo si el backend lo permite
        if _IS_INTERACTIVE_BACKEND:
            plt.ion()
        else:
            plt.ioff()

        # Configuración global de estilo
        plt.rcParams.update({
            "font.family": "Arial",   
            "font.size": 9,          
            "axes.titlesize": 10,     
            "axes.labelsize": 9,     
            "xtick.labelsize": 8,     
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }) 

        self.fig, self.ax = plt.subplots(figsize=(6.5, 4.5), dpi=100, layout="constrained")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward (MUSD)")
        self.ax.grid(True, linestyle='--', alpha=0.6)

        (self.line,) = self.ax.plot([], [], lw=1, color="#686868", alpha=0.6, label="Reward")
        (self.line_avg,) = self.ax.plot([], [], lw=2, color='#D32F2F', label=f"Moving average ({self.window})")

        self.ax.legend()
        # self.fig.show()

    def _append_reward(self, r: float) -> None:
        self.rewards_ep.append(float(r))
        w = min(self.window, len(self.rewards_ep))
        self.moving_avg.append(float(np.mean(self.rewards_ep[-w:])))

        if self.expected_total_episodes is not None:
            if len(self.rewards_ep) == int(self.expected_total_episodes):
                print(f"[INFO] Media móvil final (convergencia): {self.moving_avg[-1]:.6f}")

    def _refresh_plot(self) -> None:
        if len(self.rewards_ep) == 0: return
        if self.refresh_every <= 0: return
        if len(self.rewards_ep) % self.refresh_every != 0: return

        # Intentamos actualizar solo si la ventana existe
        try:
            x = np.arange(1, len(self.rewards_ep) + 1)
            self.line.set_data(x, self.rewards_ep)
            self.line_avg.set_data(x, self.moving_avg)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass # Ignoramos errores de GUI si la ventana se cerró o falló

    def _save_and_close(self) -> None:
        out = Path(self.filename)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Guardar CSV con los datos crudos
        csv_path = out.with_suffix(".csv")
        df = pd.DataFrame({
            "episode": range(1, len(self.rewards_ep) + 1),
            "reward": self.rewards_ep,
            "moving_avg": self.moving_avg
        })
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Datos de entrenamiento guardados en: {csv_path}")

        png_path = out.with_suffix(".png")
        try:
            self.fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"No se pudo guardar la imagen preliminar (no importa, tienes el CSV): {e}")

        plt.close(self.fig)
        plt.ioff()
        if _IS_INTERACTIVE_BACKEND:
            plt.show(block=False)
        else:
            plt.close(self.fig)

# ============================================================
# Callback para SB3 (PPO/A2C)
# ============================================================

class LivePlotCallback(BaseCallback, _LivePlotBase):
    """
    Para SB3: lee rewards por episodio desde infos -> info["episode"]["r"].
    Requiere que el env esté envuelto con Monitor/VecMonitor para que aparezca "episode".
    """
    def __init__(self, verbose: int = 0, window: int = 100, refresh_every: int = 10, title: Optional[str] = None, filename: str = "figures/sb3/training", expected_total_episodes: Optional[int] = None):
        super().__init__(verbose=verbose)
        self._init_plot(window=window, refresh_every=refresh_every, title=title, filename=filename, expected_total_episodes=expected_total_episodes)
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None and "r" in ep: self._append_reward(float(ep["r"]))
        self._refresh_plot()
        return True
    def _on_training_end(self) -> None: self._save_and_close()

class LiveRewardPlotter(_LivePlotBase):
    """
    Para Q-learning
    """
    def __init__(self, window: int = 100, refresh_every: int = 10, title: Optional[str] = None, filename: str = "figures/ql/training", expected_total_episodes: Optional[int] = None):
        self._init_plot(window=window, refresh_every=refresh_every, title=title, filename=filename, expected_total_episodes=expected_total_episodes)
    def update(self, r: float) -> None:
        self._append_reward(float(r))
        self._refresh_plot()
    def close(self) -> None: self._save_and_close()