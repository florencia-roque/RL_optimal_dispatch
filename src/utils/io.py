# src/utils/io.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type
import numpy as np
import pandas as pd

# =========================
# Lectura genérica de datos
# =========================

def leer_archivo(rutaArchivo: str | Path, sep: Optional[str] = None,
                 header: int = 0, sheet_name: int | str = 0) -> pd.DataFrame:
    """
    Lee un archivo .xlsx/.xls o .csv y lo devuelve como DataFrame.
    Parámetros:
    - rutaArchivo: Ruta al archivo a leer.
    - sep: Separador para CSV (por defecto None, que usa el separador por defecto de pandas).
    - header: Fila del encabezado (por defecto 0).
    - sheet_name: Nombre o índice de la hoja para Excel (por defecto 0).
    Retorna:
    - pd.DataFrame con los datos leídos.
    """
    rutaArchivo = str(rutaArchivo)
    if rutaArchivo.endswith((".xlsx", ".xls")):
        return pd.read_excel(rutaArchivo, header=header, sheet_name=sheet_name)
    elif rutaArchivo.endswith(".csv"):
        return pd.read_csv(rutaArchivo, sep=sep, header=header, encoding="cp1252")
    else:
        raise ValueError(f"Formato de archivo no soportado: {rutaArchivo}")

# ======================
# Utilitarios de carpetas
# ======================

def ensure_dir(path: Path) -> Path:
    """
    Crea la carpeta si no existe y devuelve el Path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

# ============
# Modelo SB3
# ============

def sb3_model_exists(model_path: Path) -> bool:
    """
    Para SB3 los modelos se guardan como <nombre>.zip.
    Recibe el path sin extensión y chequea si existe el .zip.
    """
    candidate = Path(str(model_path) + ".zip")
    return candidate.exists()

def save_sb3_model(model: Any, model_path: Path) -> None:
    """
    Guarda un modelo de Stable-Baselines3 en <path>.zip.
    """
    ensure_dir(model_path.parent)
    model.save(str(model_path))

def load_sb3_model(model_cls: Type, model_path: Path) -> Any:
    """
    Carga un modelo SB3 desde <path>.zip.
    El path se pasa sin extensión.
    """
    return model_cls.load(str(model_path))

# ==========================
# VecNormalize / normalizador
# ==========================

def save_vecnormalize(vec_normalize: Any, path: Path) -> None:
    """
    Guarda el objeto VecNormalize en un .pkl.
    """
    ensure_dir(path.parent)
    vec_normalize.save(str(path))

def load_vecnormalize(vecnorm_cls: Any, path: Path, env: Any) -> Any:
    """
    Carga VecNormalize desde .pkl y lo asocia a un env.
    """
    vecnorm = vecnorm_cls.load(str(path), env)
    return vecnorm

# ============
# Q-Table
# ============

def save_q_table(Q: np.ndarray, path: Path) -> None:
    """
    Guarda la tabla Q en formato .npy.
    """
    ensure_dir(path.parent)
    np.save(str(path), Q)

def load_q_table(path: Path) -> np.ndarray:
    """
    Carga una tabla Q desde un .npy.
    """
    return np.load(str(path))

# ==================
# Guardado de DataFrames
# ==================

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Guarda un DataFrame en CSV (utf-8, sin índice).
    """
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

def save_multiple_dataframes(dfs: Dict[str, pd.DataFrame], base_dir: Path) -> None:
    """
    Guarda varios DataFrames en la misma carpeta.
    La clave del diccionario se usa como nombre de archivo (<clave>.csv).
    """
    ensure_dir(base_dir)
    for name, df in dfs.items():
        out_path = base_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)

# ======================
# Guardado de escenarios
# ======================

def save_scenarios(dfs_escenarios: Iterable[pd.DataFrame], carpeta: Path,
                   prefix: str = "escenario_") -> None:
    """
    Guarda una lista/iterable de DataFrames, cada uno como escenario_i.csv.
    """
    ensure_dir(carpeta)
    for i, df_escenario in enumerate(dfs_escenarios):
        ruta_csv = carpeta / f"{prefix}{i}.csv"
        df_escenario.to_csv(ruta_csv, index=False)

# ======================
# Guardado de artefactos de corrida
# ======================

def save_run_artifacts(model: Any, model_path: Path,
                       vecnorm: Any | None = None,
                       vecnorm_path: Path | None = None) -> None:
    save_sb3_model(model, model_path)
    if vecnorm is not None and vecnorm_path is not None:
        save_vecnormalize(vecnorm, vecnorm_path)