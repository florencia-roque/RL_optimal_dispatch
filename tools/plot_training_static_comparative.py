# tools/plot_training_static_comparative.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

class ComparativeTrainingPlotter:
    """
    Clase para generar gráficos de entrenamiento comparativos (PPO vs Q-Learning)
    de alta calidad (CMES) a partir de archivos CSV.
    Incluye línea base de referencia (MOP).
    """
    
    def __init__(self, ppo_csv_path: str, ql_csv_path: str):
        self.ppo_csv_path = Path(ppo_csv_path)
        self.ql_csv_path = Path(ql_csv_path)
        
        if not self.ppo_csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo PPO: {ppo_csv_path}")
        if not self.ql_csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo Q-Learning: {ql_csv_path}")
            
        self.df_ppo = pd.read_csv(self.ppo_csv_path)
        self.df_ql = pd.read_csv(self.ql_csv_path)
        
        print(f"Cargados datos PPO: {self.ppo_csv_path.name} ({len(self.df_ppo)} episodios)")
        print(f"Cargados datos QL: {self.ql_csv_path.name} ({len(self.df_ql)} episodios)")

    def _get_moving_average(self, df: pd.DataFrame, window: int = 100):
        """Obtiene o calcula la media móvil de un DataFrame."""
        if 'moving_avg' in df.columns:
            return df['moving_avg']
        return df['reward'].rolling(window=window, min_periods=1).mean()

    def plot(self, reward_mop: float, window_label=100):
        """
        Genera la gráfica comparativa y guarda en PNG, PDF y TIFF.
        reward_mop: El valor de recompensa (costo negativo) del MOP para trazar la línea horizontal.
        """
        # CONFIGURACIÓN DE ESTILO (Requisitos CMES)
        plt.rcParams.update({
            "font.family": "Arial",   
            "font.size": 10,          
            "axes.titlesize": 11,     
            "axes.labelsize": 10,     
            "xtick.labelsize": 9,     
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }) 

        # Crear figura
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600, layout="constrained")
        
        # --- GRAFICAR PPO ---
        # ax.plot(
        #     self.df_ppo['episode'], 
        #     self.df_ppo['reward'], 
        #     lw=0.5, 
        #     color="#51B4E6",
        #     alpha=0.5, 
        #     zorder=1,
        #     label="PPO Reward"
        # )
        ax.plot(
            self.df_ppo['episode'], 
            self._get_moving_average(self.df_ppo, window_label), 
            lw=1.5, 
            color="#F8C395",
            label=f"PPO Moving Avg ({window_label})"
        )

        # --- GRAFICAR Q-LEARNING ---
        # ax.plot(
        #     self.df_ql['episode'], 
        #     self.df_ql['reward'], 
        #     lw=0.5, 
        #     color="#F1948A",
        #     alpha=0.5, 
        #     label="QL Reward"
        # )
        ax.plot(
            self.df_ql['episode'], 
            self._get_moving_average(self.df_ql, window_label), 
            lw=1.5, 
            color="#F19040",
            label=f"QL Moving Avg ({window_label})"
        )

        # --- LÍNEA HORIZONTAL DEL MOP ---
        ax.axhline(
            y=reward_mop, 
            color=(0.15, 0.525, 0.302), # Verde MOP
            linestyle="--", 
            linewidth=1.5, 
            label="MOP Baseline"
        )

        # Formato de los ejes
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (MUSD)")
        ax.set_ylim(-5000, -2000)
        ax.grid(True, linestyle='--', alpha=0.6)
    
        ax.legend(loc="best", framealpha=0.9)
        
        # Guardar resultados
        output_dir = self.ppo_csv_path.parent / "comparative_plots"
        output_dir.mkdir(exist_ok=True)
        
        base_name = "training_comparison_ppo_vs_ql"
        png_path = output_dir / f"{base_name}.png"
        pdf_path = output_dir / f"{base_name}.pdf"
        tiff_path = output_dir / f"{base_name}.tif"

        print(f"Guardando figuras en: {output_dir}")

        fig.savefig(str(png_path), dpi=600, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=0.02) # Vectorial

        # TIFF con compresión LZW
        fig.savefig(
            str(tiff_path), 
            dpi=600, 
            format="tiff", 
            facecolor='white', 
            bbox_inches="tight",
            pad_inches=0.02,
            pil_kwargs={"compression": "tiff_lzw"}
        )
        
        plt.close(fig)
        
        # POST-PROCESAMIENTO PIL
        self._post_process_tiff(tiff_path)

    def _post_process_tiff(self, tiff_path: Path):
        """Aplica las correcciones de color y tamaño requeridas."""
        try:
            img = Image.open(tiff_path)

            if img.mode != 'RGB':
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                img = background

            target_width_inch = 6.5
            dpi = 600
            target_width_px = int(target_width_inch * dpi)
            current_width_px = img.size[0]

            if abs(target_width_px - current_width_px) > (target_width_px * 0.01):
                aspect_ratio = img.size[1] / img.size[0]
                target_height_px = int(target_width_px * aspect_ratio)
                img = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

            img.save(tiff_path, dpi=(600, 600), compression="tiff_lzw")
            print("Imagen TIFF optimizada guardada exitosamente.")
            
        except Exception as e:
            print(f"[ERROR] Falló el post-procesamiento de imagen: {e}")


if __name__ == "__main__":
    REWARD_MOP = -2350

    root = tk.Tk()
    root.withdraw()
    
    print("Paso 1: Selecciona el archivo CSV de PPO...")
    ppo_path = filedialog.askopenfilename(
        title="Selecciona el CSV de PPO",
        filetypes=[("Archivos CSV", "*.csv")]
    )
    
    if not ppo_path:
        print("Operación cancelada.")
        exit()

    print("Paso 2: Selecciona el archivo CSV de Q-Learning...")
    ql_path = filedialog.askopenfilename(
        title="Selecciona el CSV de Q-Learning",
        filetypes=[("Archivos CSV", "*.csv")]
    )
    
    if ppo_path and ql_path:
        plotter = ComparativeTrainingPlotter(ppo_path, ql_path)
        plotter.plot(reward_mop=REWARD_MOP, window_label=100) 
        print("\nProceso finalizado.")
    else:
        print("Se requieren ambos archivos para comparar.")