# src/utils/callbacks.py

from __future__ import annotations
from pathlib import Path
from PIL import Image
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

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

        # Interactivo
        plt.ion()

        # Configuración global de estilo
        # Requisitos CMES
        plt.rcParams.update({
            "font.family": "Arial",   
            "font.size": 9,          
            "axes.titlesize": 10,     
            "axes.labelsize": 9,     
            "xtick.labelsize": 8,     
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }) 

        self.fig, self.ax = plt.subplots(figsize=(6.5, 4.5), dpi=600, layout="constrained")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward (MUSD)")
        
        # self.ax.set_title(title or "", pad=8)
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Gráfica "ruidosa" (Reward instantáneo): Color gris
        (self.line,) = self.ax.plot(
            [], [], 
            lw=1, 
            color="#686868",  
            alpha=0.6,       
            label="Reward"
        )
        
        # Gráfica "tendencia" (Media móvil): Color rojo oscuro
        (self.line_avg,) = self.ax.plot(
            [], [], 
            lw=2,          
            color='#D32F2F',  
            label=f"Moving average ({self.window})"
        )

        self.ax.legend()
        self.fig.show()

    def _append_reward(self, r: float) -> None:
        self.rewards_ep.append(float(r))
        w = min(self.window, len(self.rewards_ep))
        self.moving_avg.append(float(np.mean(self.rewards_ep[-w:])))

        if self.expected_total_episodes is not None:
            if len(self.rewards_ep) == int(self.expected_total_episodes):
                print(
                    f"[INFO] Media móvil final (convergencia): {self.moving_avg[-1]:.6f}"
                )

    def _refresh_plot(self) -> None:
        if len(self.rewards_ep) == 0:
            return
        if self.refresh_every <= 0:
            return
        if len(self.rewards_ep) % self.refresh_every != 0:
            return

        x = np.arange(1, len(self.rewards_ep) + 1)
        self.line.set_data(x, self.rewards_ep)
        self.line_avg.set_data(x, self.moving_avg)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _save_and_close(self) -> None:
        out = Path(self.filename)
        out.parent.mkdir(parents=True, exist_ok=True)

        png_path = out
        if png_path.suffix == "":
            png_path = png_path.with_suffix(".png")

        pdf_path = png_path.with_suffix(".pdf")

        tiff_path = out.with_suffix(".tif")

        self.fig.savefig(str(png_path), dpi=600, bbox_inches="tight", pad_inches=0)
        self.fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=0)  # vectorial para el paper

        # compression='tiff_lzw': Recomendado para que el archivo no pese 100MB (sin perder calidad)
        self.fig.savefig(
            str(tiff_path), 
            dpi=600, 
            format="tiff", 
            facecolor='white', 
            transparent=False,
            bbox_inches="tight",
            pad_inches=0,
            pil_kwargs={"compression": "tiff_lzw"}
        )

        # Abrir la imagen que matplotlib generó mal
        img = Image.open(tiff_path)

        # CORREGIR MODO DE COLOR (Forzar RGB puro)
        if img.mode != 'RGB':
            print(f"Corrigiendo modo de color: {img.mode} -> RGB")
            background = Image.new("RGB", img.size, (255, 255, 255)) # Fondo blanco
            background.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None) # Pegar encima
            img = background

        # CORREGIR TAMAÑO (Reescalar a 6.5 pulgadas exactas si es necesario)
        target_width_inch = 6.5
        dpi = 600
        target_width_px = int(target_width_inch * dpi)
        current_width_px = img.size[0]

        # Solo reescalar si la diferencia es notable (>5%)
        if abs(target_width_px - current_width_px) > (target_width_px * 0.05):
            print(f"Corrigiendo tamaño: {current_width_px}px -> {target_width_px}px (Ancho 6.5\")")
            aspect_ratio = img.size[1] / img.size[0]
            target_height_px = int(target_width_px * aspect_ratio)
            
            # Reescalado de alta calidad (LANCZOS)
            img = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

        # Guardar la versión FINAL CORREGIDA (Sobrescribir)
        img.save(
            tiff_path,
            dpi=(600, 600),
            compression="tiff_lzw"
        )
        print("Imagen corregida y guardada exitosamente (RGB, 6.5\", 600 DPI).")
        # ---------------------------------------

        plt.ioff()
        # no bloquea en ejecuciones no-interactivas
        plt.show(block=False)

# ============================================================
# Callback para SB3 (PPO/A2C)
# ============================================================

class LivePlotCallback(BaseCallback, _LivePlotBase):
    """
    Para SB3: lee rewards por episodio desde infos -> info["episode"]["r"].
    Requiere que el env esté envuelto con Monitor/VecMonitor para que aparezca "episode".
    """

    def __init__(
        self,
        verbose: int = 0,
        window: int = 100,
        refresh_every: int = 10,
        title: Optional[str] = None,
        filename: str = "figures/sb3/training",
        expected_total_episodes: Optional[int] = None,
    ):
        super().__init__(verbose=verbose)
        self._init_plot(
            window=window,
            refresh_every=refresh_every,
            title=title,
            filename=filename,
            expected_total_episodes=expected_total_episodes,
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None and "r" in ep:
                self._append_reward(float(ep["r"]))

        self._refresh_plot()
        return True

    def _on_training_end(self) -> None:
        self._save_and_close()

# ============================================================
# Plotter para Q-learning (manual)
# ============================================================

class LiveRewardPlotter(_LivePlotBase):
    """
    Para Q-learning
    """

    def __init__(
        self,
        window: int = 100,
        refresh_every: int = 10,
        title: Optional[str] = None,
        filename: str = "figures/ql/training",
        expected_total_episodes: Optional[int] = None,
    ):
        self._init_plot(
            window=window,
            refresh_every=refresh_every,
            title=title,
            filename=filename,
            expected_total_episodes=expected_total_episodes,
        )

    def update(self, r: float) -> None:
        self._append_reward(float(r))
        self._refresh_plot()

    def close(self) -> None:
        self._save_and_close()