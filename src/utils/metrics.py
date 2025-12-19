# src/utils/metrics.py
"""
Módulo de métricas para evaluar resultados de RL
en el problema hidro–térmico.

Recibe `df_avg` y `df_all` generados por evaluator_sb3 o qlearning_agent.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# MÉTRICAS BASE
# ============================================================

def total_reward(df_avg: pd.DataFrame) -> float:
    """Reward total acumulado del episodio promedio."""
    return df_avg["reward"].sum()


def mean_reward(df_avg: pd.DataFrame) -> float:
    """Reward medio por paso de tiempo."""
    return df_avg["reward"].mean()


def reward_std(df_all: pd.DataFrame) -> float:
    """Desvío estándar del reward paso a paso (todos los episodios)."""
    return df_all["reward"].std()


# ============================================================
# ENERGÍA
# ============================================================

ENERGY_COLS = [
    "energia_hidro",
    "energia_eolica",
    "energia_solar",
    "energia_biomasa",
    "energia_renovable",
    "energia_termico_bajo",
    "energia_termico_alto",
]

def total_energy(df_avg: pd.DataFrame) -> pd.Series:
    """Energía total generada por cada fuente."""
    return df_avg[ENERGY_COLS].sum()


def renewable_share(df_avg: pd.DataFrame) -> float:
    """Porcentaje de energía renovable sobre el total."""
    total = (
        df_avg["energia_renovable"].sum()
        + df_avg["energia_termico_bajo"].sum()
        + df_avg["energia_termico_alto"].sum()
    )
    if total == 0:
        return np.nan
    return df_avg["energia_renovable"].sum() / total


def thermal_ratio(df_avg: pd.DataFrame) -> float:
    """Fracción de energía térmica sobre el total."""
    total = (
        df_avg["energia_termico_bajo"].sum()
        + df_avg["energia_termico_alto"].sum()
        + df_avg["energia_renovable"].sum()
    )
    if total == 0:
        return np.nan
    thermal = (
        df_avg["energia_termico_bajo"].sum()
        + df_avg["energia_termico_alto"].sum()
    )
    return thermal / total


def peak_thermal(df_avg: pd.DataFrame) -> float:
    """Máxima generación térmica por paso de tiempo."""
    thermal = df_avg["energia_termico_bajo"] + df_avg["energia_termico_alto"]
    return thermal.max()


def mean_hydro(df_avg: pd.DataFrame) -> float:
    """Promedio de generación hidráulica."""
    return df_avg["energia_hidro"].mean()


# ============================================================
# HIDRÁULICA / EMBALSE
# ============================================================

def mean_volume(df_avg: pd.DataFrame) -> float:
    """Volumen promedio del embalse."""
    return df_avg["volumen"].mean() if "volumen" in df_avg.columns else np.nan


def spill_total(df_avg: pd.DataFrame) -> float:
    """Total de vertimientos."""
    return df_avg.get("vertimiento", pd.Series([0])).sum()


def spill_frequency(df_avg: pd.DataFrame) -> int:
    """Cantidad de pasos donde el vertimiento > 0."""
    if "vertimiento" not in df_avg.columns:
        return 0
    return (df_avg["vertimiento"] > 0).sum()


def average_inflows(df_avg: pd.DataFrame) -> float:
    """Promedio de aportes hídricos."""
    return df_avg["aportes"].mean()


# ============================================================
# COSTOS
# ============================================================

def total_cost(df_avg: pd.DataFrame) -> float:
    """Costo térmico total del episodio promedio."""
    return df_avg["costo_termico"].sum()


def export_income(df_avg: pd.DataFrame) -> float:
    """Ingreso por exportación total."""
    return df_avg["ingreso_exportacion"].sum()


def net_cost(df_avg: pd.DataFrame) -> float:
    """Costo neto = costo térmico - ingresos por exportación."""
    return (
        df_avg["costo_termico"].sum()
        - df_avg["ingreso_exportacion"].sum()
    )


# ============================================================
# ACCIONES DEL AGENTE
# ============================================================

def action_variability(df_all: pd.DataFrame) -> float:
    """Desvío estándar de la acción (proporción turbinada)."""
    return df_all["action"].std()


def mean_action(df_all: pd.DataFrame) -> float:
    """Acción promedio tomada."""
    return df_all["action"].mean()


def policy_smoothness(df_all: pd.DataFrame) -> float:
    """
    Suavidad de la política:
    std de la diferencia entre acciones consecutivas.
    """
    a = df_all["action"].values
    if len(a) < 2:
        return np.nan
    return np.diff(a).std()


# ============================================================
# DEMANDA Y RESIDUAL
# ============================================================

def mean_residual(df_avg: pd.DataFrame) -> float:
    """Promedio de demanda residual."""
    return df_avg["demanda_residual"].mean()


def unmet_demand_events(df_all: pd.DataFrame) -> int:
    """
    Cantidad de pasos donde renovables + hidráulica + térmica < demanda.
    (Solo si el entorno lo reporta en residual)
    """
    return (df_all["demanda_residual"] > 0).sum()


# ============================================================
# PAQUETE DE MÉTRICAS COMPLETO
# ============================================================

def compute_all_metrics(df_avg: pd.DataFrame, df_all: pd.DataFrame) -> dict:
    """
    Devuelve un diccionario con TODAS las métricas importantes.
    Ideal para comparar PPO vs A2C vs QL.
    """

    metrics = {
        # Reward
        "reward_total": total_reward(df_avg),
        "reward_mean": mean_reward(df_avg),
        "reward_std": reward_std(df_all),

        # Energía
        "total_energy": total_energy(df_avg).to_dict(),
        "renewable_share": renewable_share(df_avg),
        "thermal_ratio": thermal_ratio(df_avg),
        "peak_thermal": peak_thermal(df_avg),
        "hydro_mean": mean_hydro(df_avg),

        # Embalse
        "volume_mean": mean_volume(df_avg),
        "spill_total": spill_rate(df_avg),
        "spill_frequency": spill_frequency(df_avg),
        "average_inflows": average_inflows(df_avg),

        # Costos
        "cost_total": total_cost(df_avg),
        "export_income": export_income(df_avg),
        "net_cost": net_cost(df_avg),

        # Política del agente
        "action_mean": mean_action(df_all),
        "action_std": action_variability(df_all),
        "policy_smoothness": policy_smoothness(df_all),

        # Demanda
        "residual_mean": mean_residual(df_avg),
        "unmet_demand_events": unmet_demand_events(df_avg),
    }

    return metrics
