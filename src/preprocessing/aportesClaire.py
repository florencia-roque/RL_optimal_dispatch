import pandas as pd
import numpy as np

from src.utils.config import (
    CLAIRE_XLT,
    CLAIRE_APORTE_CSV,
    CLAIRE_HIDROLOGIA_CSV,
    CLAIRE_MATRICES_CSV,
)

def main():
    print("Procesando datos de aportes de Claire...")

    # 1) Levantar datos históricos crudos
    df = pd.read_csv(
        CLAIRE_XLT,
        sep=r"\s+",
        header=7,
        encoding="cp1252"
    )

    # 2) Calcular aporte total de Claire como suma de Bonete + Palmar + Salto
    df["APORTE-CLAIRE"] = df[["APORTE-BONETE", "APORTE-PALMAR", "APORTE-SALTO"]].sum(axis=1)

    # Renombrar la columna Estacion → Semana
    df = df.rename(columns={"Estacion": "Semana"})

    # Selección de columnas relevantes
    nuevo_df = df[["Cronica", "Semana", "APORTE-CLAIRE"]]

    # Pivotear: filas = semanas, columnas = crónicas
    pivot_df = nuevo_df.pivot(index="Semana", columns="Cronica", values="APORTE-CLAIRE")

    # Guardar aporte_claire.csv
    pivot_df.to_csv(CLAIRE_APORTE_CSV, index=False)
    print(f"Aporte medio semanal guardado en: {CLAIRE_APORTE_CSV}")

    # 3) Clasificación por quintiles
    n_clases = 5
    df_clasificado = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)

    for semana_idx, fila in pivot_df.iterrows():
        valores = pd.to_numeric(fila, errors="coerce")

        try:
            clases = pd.qcut(valores, q=n_clases, labels=list(range(n_clases)))
        except ValueError:
            clases = pd.Series([None] * len(valores), index=valores.index)

        df_clasificado.loc[semana_idx] = clases

    # Guardar hidrologia_claire.csv
    df_clasificado.to_csv(CLAIRE_HIDROLOGIA_CSV, index=False)
    print(f"Clasificación hidrológica guardada en: {CLAIRE_HIDROLOGIA_CSV}")

    # 4) Preparar datos para matrices de transición semanales
    df_numeric = df_clasificado.replace({i: i for i in range(n_clases)}).dropna()

    # Rotar una fila para comparar semana 52 → semana 1 siguiente
    primer_fila = df_numeric.iloc[0].tolist()
    rotado = primer_fila.pop(0)
    primer_fila.append(rotado)

    df_rotado = pd.concat(
        [df_numeric, pd.DataFrame([primer_fila], columns=df_numeric.columns)],
        ignore_index=True
    )

    # 5) Calcular matrices de transición semana a semana
    def matriz_transicion(origen, destino, clases):
        matriz = np.zeros((clases, clases), dtype=int)
        for o, d in zip(origen, destino):
            if pd.notna(o) and pd.notna(d):
                matriz[int(o), int(d)] += 1
        matriz_pct = matriz / matriz.sum(axis=1, keepdims=True)
        return matriz_pct

    filas = []
    semanas = df_rotado.index.tolist()

    for i in range(len(semanas) - 1):
        origen = df_rotado.loc[i].values
        destino = df_rotado.loc[i + 1].values
        matriz_pct = matriz_transicion(origen, destino, n_clases)
        fila = [i] + matriz_pct.flatten().tolist()
        filas.append(fila)

    columnas = ["Semana"] + [f"{i}-{j}" for i in range(n_clases) for j in range(n_clases)]
    df_matrices = pd.DataFrame(filas, columns=columnas)

    # Guardar matrices_markov_claire.csv
    df_matrices.to_csv(CLAIRE_MATRICES_CSV, index=False)
    print(f"Matrices de transición de Markov guardadas en: {CLAIRE_MATRICES_CSV}")

    print("Proceso finalizado con éxito.")


if __name__ == "__main__":
    main()