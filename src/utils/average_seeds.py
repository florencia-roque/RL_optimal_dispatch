import pandas as pd
from pathlib import Path
import re
import os

from pathlib import Path

# Escenarios que querés promediar
escenarios = [4, 7, 108]

# Carpeta donde están TODOS los archivos


carpeta = Path(
    r"C:\Users\ut605453\Downloads\eval_2026-01-22_20-10-20_est_markov"
)

def leer_archivo(archivo):
    return pd.read_csv(
        archivo,
        sep="\t",
        header=0,
        encoding="latin1",
        comment="@",
        engine="python"
    )

# Columnas a promediar
columnas_objetivo = [
    "energia_hidro",
    "energia_renovable",
    "energia_termico_bajo",
    "energia_termico_alto",
    "demanda",
    "aportes",
    "volumen",
]

dfs = []


for esc in escenarios:
    dfs = []  # acumulador POR ESCENARIO

    for directorio in os.listdir(carpeta):
        subcarpeta = carpeta / directorio
        if not subcarpeta.is_dir():
            continue

        for archivo in subcarpeta.glob("*.csv"):
            match = re.search(r"escenario_(\d+)", archivo.stem)
            if not match:
                continue

            esc_archivo = int(match.group(1))
            if esc_archivo != esc:
                continue

            df = pd.read_csv(archivo)
            dfs.append(df)

    if not dfs:
        print(f"No se encontraron archivos para escenario {esc}")
        continue

    df_total = pd.concat(dfs, ignore_index=True)

    # Promedio del escenario entre carpetas
    df_promedio = (
        df_total
        .groupby("tiempo", as_index=False)[columnas_objetivo]
        .mean()
    )

    # Guardar CSV por escenario
    salida = carpeta / f"escenario_{esc}_promedio.csv"
    df_promedio.to_csv(salida, index=False)


""""

# Iterar todos los archivos escenario_*.csv
for directorio in os.listdir(carpeta):
    subcarpeta = carpeta / directorio
 
    for archivo in carpeta.rglob("*.csv"):
        archivo = Path(archivo)   # no hace daño si ya es Path
        match = re.search(r"escenario_(\d+)", archivo.stem)
        if not match:
            continue

        esc = int(match.group(1))

        # Filtrar solo los escenarios deseados
        if esc not in escenarios:
            continue

        df = pd.read_csv(archivo)
        df["escenario"] = esc
        dfs.append(df)

        # Unir solo los escenarios seleccionados
        df_total = pd.concat(dfs, ignore_index=True)

        # Promedio entre escenarios por tiempo
        df_promedio = (
        df_total
        .groupby("tiempo")[columnas_objetivo]
        .mean()
        .reset_index()
        )

        # Guardar resultado
        df_promedio.to_csv(subcarpeta / "escenario_promedio.csv", index=False)


"""

