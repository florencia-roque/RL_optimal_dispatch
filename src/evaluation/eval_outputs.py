# src/evaluation/eval_outputs.py

from __future__ import annotations
import pandas as pd
from src.utils.paths import evaluation_dirs, evaluation_csv_paths
from src.utils.io import save_dataframe, save_scenarios

def save_eval_outputs(
    df_avg: pd.DataFrame,
    df_all: pd.DataFrame,
    *,
    alg: str,
    fecha: str,
    mode_tag_str: str,
    estados_cols: list[str],
    n_eval_episodes: int | None = None,
):
    """
    Guarda TODOS los outputs de evaluación:
    - trayectorias (df_avg)
    - energías, estados, resultados_agente, costos
    - escenarios individuales (df_all agrupado por episode_id)

    `estados_cols` depende del env:
      - SB3 continuo: ["volumen", ...]
      - QL tabular:  ["volumen_discreto", ...]
    """
    
    # Directorios / paths
    dirs = evaluation_dirs(alg, fecha, mode_tag_str)
    eval_dir = dirs["eval_dir"]
    promedios_dir = dirs["promedios_dir"]
    csv_paths = evaluation_csv_paths(promedios_dir)

    # reward_usd (si existe reward)
    if "reward" in df_avg.columns and "reward_usd" not in df_avg.columns:
        df_avg = df_avg.copy()
        df_avg["reward_usd"] = df_avg["reward"] * 1e6

    # Trayectorias completas
    save_dataframe(df_avg, csv_paths["trayectorias"])

    # Energías (si están)
    columnas_energias = [
        "energia_hidro",
        "energia_eolica",
        "energia_solar",
        "energia_biomasa",
        "energia_renovable",
        "energia_termico_bajo",
        "energia_termico_alto",
        "demanda",
        "demanda_residual",
    ]
    cols_ok = [c for c in columnas_energias if c in df_avg.columns]
    if cols_ok:
        save_dataframe(df_avg[cols_ok], csv_paths["energias"])

    # Estados
    cols_ok = [c for c in estados_cols if c in df_avg.columns]
    if cols_ok:
        save_dataframe(df_avg[cols_ok], csv_paths["estados"])

    # Resultados del agente
    columnas_agente = ["action", "fraccion_turbinado", "reward"]
    cols_ok = [c for c in columnas_agente if c in df_avg.columns]
    if cols_ok:
        save_dataframe(df_avg[cols_ok], csv_paths["resultados_agente"])

    # Costos
    columnas_costos = ["costo_termico", "ingreso_exportacion"]
    cols_ok = [c for c in columnas_costos if c in df_avg.columns]
    if cols_ok:
        save_dataframe(df_avg[cols_ok], csv_paths["costos"])

    # Escenarios individuales
    if "episode_id" in df_all.columns:
        dfs_escenarios = [
            g.reset_index(drop=True)
            for _, g in df_all.groupby("episode_id", sort=True)
        ]
        if n_eval_episodes is not None:
            dfs_escenarios = dfs_escenarios[:n_eval_episodes]
        save_scenarios(dfs_escenarios, eval_dir)

    # reward total
    reward_total = float(df_avg["reward"].sum()) if "reward" in df_avg.columns else float("nan")
    return reward_total, dirs, csv_paths