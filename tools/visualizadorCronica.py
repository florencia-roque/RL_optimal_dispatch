# tools/visualizadorCronica.py

import os
from matplotlib import ticker
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Image, Tk, filedialog
from PIL import Image
import unicodedata

# Función para normalizar nombres de columnas
def normalizar_columna(col):
    col = col.lower().replace(" ", "_")
    col = ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')
    return col

# Abrir file chooser
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Selecciona el archivo CSV",
    filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
)
if not file_path:
    print("No se seleccionó ningún archivo.")
    exit()

# Leer CSV y normalizar columnas
if file_path.endswith('csv'):
    df = pd.read_csv(file_path, sep=",", engine="python")
    df.columns = [normalizar_columna(c) for c in df.columns]
    print("Columnas normalizadas:", df.columns.tolist())

elif file_path.endswith('xlsx'):
    df = pd.read_excel(file_path,header=0)
    df.columns = [normalizar_columna(c) for c in df.columns]
    print("Columnas normalizadas:", df.columns.tolist())

# Definir columnas
col_turbinada = "energia_hidro"
col_renovable = "energia_renovable"
col_termico_bajo = "energia_termico_bajo"
col_termico_alto = "energia_termico_alto"
col_demanda = "demanda"
col_aportes = "aportes"
col_volumen = "volumen"

# Eje X
x = df.index

# Configuración global de estilo
# Requisitos CMES
plt.rcParams.update({
    "font.family": "Arial",   
    "font.size": 10,          
    "axes.titlesize": 11,     
    "axes.labelsize": 10,     
    "xtick.labelsize": 9,     
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})        

# Crear figura alta resolución
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600, layout="constrained")

# Áreas apiladas
ax.stackplot(
    x,
    df[col_turbinada]/1000,     # Hydro (azul por defecto)
    df[col_termico_bajo]/1000,  # Thermal low-cost (naranja)
    df[col_termico_alto]/1000,  # Thermal high-cost (verde)
    labels=["Hydro", "Thermal low-cost", "Thermal high-cost"],
    colors=["#b8fff6", "#fbd1e7", "#ff0000"]  # turquesa, rosa, rojo
)

# Demanda (negro)
ax.plot(x, df[col_demanda]/1000, label="Demand", color="#000000", linewidth=2.4)

# Eje secundario
ax2 = ax.twinx()

ax.set_xlim(x.min(), x.max())

# Inflows: marrón punteado
ax2.plot(x, df[col_aportes], label="Inflows", color="#25802B", linestyle="--", linewidth=2.4)

# Reservoir volume: **naranja continua**
ax2.plot(x, df[col_volumen], label="Reservoir volume", color="#220eff", linestyle="-", linewidth=2.4)

# Títulos y etiquetas (en inglés)
# ax.set_title("Out-of-sample policy performance under historical chronicle evaluation", pad=8)
ax.set_xlabel("Week", labelpad=8)
ax.set_ylabel("Energy (GWh)", labelpad=8)

# Etiqueta del eje derecho más chica y con espacio extra para que no se corte
ax2.set_ylabel("Volume (hm³) - Inflows (hm³/week)", labelpad=8)

# Leyenda combinada, un poco más abajo para no chocar con 'Week'
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
 
# Layout: más margen derecho y abajo para etiquetas y leyenda
plt.tight_layout(rect=[0, 0, 0.98, 1])  # deja 2% libre a la derecha
fig.subplots_adjust(right=0.89, bottom=0.28)

# Guardado
alg = "ql" # Cambiar según el algoritmo usado
modo_eval = "Deterministico" # Historico o Markov o Deterministico
final_nombre = "det"

os.makedirs(f"results/figures/{alg}/chronicles/{modo_eval}", exist_ok=True)
plt.savefig(f"results/figures/{alg}/chronicles/{modo_eval}/dispatch_evaluation_{final_nombre}.pdf", bbox_inches="tight")  # vectorial para el paper

tiff_path = f"results/figures/{alg}/chronicles/{modo_eval}/dispatch_evaluation_{final_nombre}.tif"

# compression='tiff_lzw': Recomendado para que el archivo no pese 100MB (sin perder calidad)
plt.savefig(
    str(tiff_path), 
    dpi=600, 
    format="tiff", 
    facecolor='white', 
    transparent=False,
    bbox_inches=None,
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
plt.show()