import pandas as pd
from pathlib import Path
import re
import os

from pathlib import Path
from src.utils import config


class AverageSeeds:
    def main(algoritmo):


        # Escenarios que querés promediar
        escenarios = [4, 7, 108]

        # Carpeta donde están TODOS los archivos

        
        carpeta = config.EVALUATIONS / algoritmo

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
                print("RUTA: ")
                print( subcarpeta)
                
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