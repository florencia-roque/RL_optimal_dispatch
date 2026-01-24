# tools/plot_training_static.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

class StaticTrainingPlotter:
    """
    Clase para generar gráficos de entrenamiento de alta calidad (CMES)
    a partir de archivos CSV generados por los callbacks.
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        print(f"Cargados datos de: {self.csv_path.name} ({len(self.df)} episodios)")

    def plot(self, window_label=None):
        """
        Genera la gráfica y guarda en formatos PNG, PDF y TIFF con alta calidad.
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
        
        # Graficar datos
        # Reward instantáneo (Gris)
        ax.plot(
            self.df['episode'], 
            self.df['reward'], 
            lw=0.8, # Línea un poco más fina para estático se ve mejor
            color="#686868", 
            alpha=0.5, 
            label="Reward"
        )
        
        # Media Móvil (Rojo Oscuro)
        # Usamos la columna 'moving_avg' si existe, si no la recalculamos
        if 'moving_avg' in self.df.columns:
            y_avg = self.df['moving_avg']
            win_txt = window_label if window_label else "Avg"
        else:
            # Fallback por si el CSV es viejo
            win = 100
            y_avg = self.df['reward'].rolling(window=win, min_periods=1).mean()
            win_txt = f"Moving average ({win})"

        ax.plot(
            self.df['episode'], 
            y_avg, 
            lw=1.5, 
            color='#D32F2F', 
            label=win_txt if window_label is None else f"Moving average ({window_label})"
        )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (MUSD)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        # Guardamos en la misma carpeta que el CSV, subcarpeta 'high_res'
        output_dir = self.csv_path.parent / "high_res"
        output_dir.mkdir(exist_ok=True)
        
        base_name = self.csv_path.stem
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

            # Corregir modo de color a RGB
            if img.mode != 'RGB':
                print(f"Corrigiendo modo de color: {img.mode} -> RGB")
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                img = background

            # Corregir tamaño a 6.5 pulgadas exactas
            target_width_inch = 6.5
            dpi = 600
            target_width_px = int(target_width_inch * dpi)
            current_width_px = img.size[0]

            if abs(target_width_px - current_width_px) > (target_width_px * 0.01):
                print(f"Corrigiendo tamaño: {current_width_px}px -> {target_width_px}px")
                aspect_ratio = img.size[1] / img.size[0]
                target_height_px = int(target_width_px * aspect_ratio)
                img = img.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

            # Guardar versión final
            img.save(tiff_path, dpi=(600, 600), compression="tiff_lzw")
            print(f"Imagen TIFF optimizada guardada exitosamente.")
            
        except Exception as e:
            print(f"[ERROR] Falló el post-procesamiento de imagen: {e}")

if __name__ == "__main__":
    # Selector de archivo simple
    root = tk.Tk()
    root.withdraw()
    
    print("Selecciona el archivo CSV de entrenamiento...")
    file_path = filedialog.askopenfilename(
        title="Selecciona el CSV de entrenamiento (training.csv)",
        filetypes=[("Archivos CSV", "*.csv")]
    )
    
    if file_path:
        plotter = StaticTrainingPlotter(file_path)
        plotter.plot(window_label=100) 
        print("\nProceso finalizado.")
    else:
        print("No se seleccionó archivo.")