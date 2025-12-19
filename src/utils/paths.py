# src/utils/paths.py

from __future__ import annotations

from typing import Tuple, Optional

from datetime import datetime
from pathlib import Path
from typing import Dict

from src.utils.config import FIGURES, MODELS, EVALUATIONS


def _normalize_alg(alg: str) -> str:
    a = alg.strip().lower()
    if a in {"qlearning", "q_learning", "q-learning", "ql"}:
        return "ql"
    if a in {"ppo"}:
        return "ppo"
    if a in {"a2c"}:
        return "a2c"
    return a

# ==================
# Utilitarios generales
# ==================

def timestamp() -> str:
    """
    Devuelve un timestamp estándar para nombres de archivos.
    Ejemplo: 2025-11-19_10-30-00
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def mode_tag(deterministico: int, modo: str | None) -> str:
    """
    Construye el tag de modo que venías usando:
    - est_<modo>  si DETERMINISTICO == 0
    - det_<modo>  si DETERMINISTICO == 1
    Si modo es None, devuelve solo 'est' o 'det'.
    """
    pref = "est" if deterministico == 0 else "det"
    if (modo is None):
        return f"{pref}"
    else:
        return f"{pref}_{modo}"

# =======================
# Rutas de entrenamiento
# =======================

_MODEL_BASENAME = {
    "ppo": "RecurrentPPO_hydro_thermal_claire_continuous",
    "a2c": "a2c_hydro_thermal_claire",
    "ql": "Q_table",
}


def training_dirs(alg: str) -> Dict[str, Path]:
    """
    Devuelve las carpetas base de entrenamiento para un algoritmo.
    keys: 'fig_dir', 'model_dir'
    """
    fig_dir = FIGURES / alg
    model_dir = MODELS / alg
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return {"fig_dir": fig_dir, "model_dir": model_dir}


def training_paths(alg: str, fecha_hora: str, mode_tag_str: str) -> Dict[str, Path]:
    """
    Devuelve paths de entrenamiento para un algoritmo dado:
    - fig_path: figura del entrenamiento
    - model_path: modelo (SB3 o Q-table)
    - vecnorm_path (solo para PPO, si aplica)
    """
    dirs = training_dirs(alg)
    fig_dir = dirs["fig_dir"]
    model_dir = dirs["model_dir"]

    fig_path = fig_dir / f"train_{fecha_hora}_{mode_tag_str}"

    base = _MODEL_BASENAME.get(alg, alg)
    if alg in {"ql", "qlearning"}:
        model_path = model_dir / f"{base}_{fecha_hora}_{mode_tag_str}.npy"
        vecnorm_path = None
    elif alg == "ppo":
        model_path = model_dir / f"{base}_{fecha_hora}_{mode_tag_str}"
        vecnorm_path = model_dir / f"vecnorm_{fecha_hora}_{mode_tag_str}.pkl"
    else:  # a2c u otros
        model_path = model_dir / f"{base}_{fecha_hora}_{mode_tag_str}"
        vecnorm_path = None

    out: Dict[str, Path] = {
        "fig_path": fig_path,
        "model_path": model_path,
    }
    if vecnorm_path is not None:
        out["vecnorm_path"] = vecnorm_path
    return out


# =======================
# Rutas de evaluación
# =======================

def evaluation_dirs(alg: str, fecha_hora: str, mode_tag_str: str) -> Dict[str, Path]:
    """
    Devuelve las carpetas base para evaluación:
    - eval_dir: carpeta de la corrida de evaluación
    - promedios_dir: subcarpeta 'promedios'
    """
    eval_dir = EVALUATIONS / alg / f"eval_{fecha_hora}_{mode_tag_str}"
    promedios_dir = eval_dir / "promedios"

    eval_dir.mkdir(parents=True, exist_ok=True)
    promedios_dir.mkdir(parents=True, exist_ok=True)

    return {"eval_dir": eval_dir, "promedios_dir": promedios_dir}


def evaluation_csv_paths(promedios_dir: Path) -> Dict[str, Path]:
    """
    Devuelve los paths de los CSV estándar de evaluación.
    Coinciden con los nombres que ya venías usando.
    """
    return {
        "trayectorias": promedios_dir / "trayectorias.csv",
        "energias": promedios_dir / "energias.csv",
        "estados": promedios_dir / "estados.csv",
        "resultados_agente": promedios_dir / "resultados_agente.csv",
        "costos": promedios_dir / "costos.csv",
    }

def get_latest_model(alg: str) -> tuple[Path, Optional[Path]]:
    """
    Devuelve:
      - model_path: path del modelo (para SB3: SIN .zip; para QL: .npy)
      - vecnorm_path: solo para PPO (si existe), sino None
    """
    alg_n = _normalize_alg(alg)

    alg_dir = MODELS / alg_n
    if not alg_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de modelos para {alg_n}")

    if alg_n == "ql":
        modelos = sorted(alg_dir.glob("*.npy"), key=lambda p: p.stat().st_mtime)
        if not modelos:
            raise FileNotFoundError(f"No hay Q-tables guardadas para {alg_n}")
        return modelos[-1], None

    # SB3: buscamos zip y devolvemos el path base (sin .zip)
    modelos_zip = sorted(alg_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not modelos_zip:
        raise FileNotFoundError(f"No hay modelos SB3 guardados para {alg_n}")

    latest_zip = modelos_zip[-1]
    model_path = latest_zip.with_suffix("")  # quita .zip

    vecnorm_path = None
    if alg_n == "ppo":
        # ideal: vecnorm con el mismo timestamp/tag
        # fallback: el más nuevo
        candidates = list(alg_dir.glob(f"vecnorm_{latest_zip.stem.replace(_MODEL_BASENAME.get('ppo','ppo')+'_','')}.pkl"))
        if candidates:
            vecnorm_path = candidates[0]
        else:
            vecs = sorted(alg_dir.glob("vecnorm_*.pkl"), key=lambda p: p.stat().st_mtime)
            vecnorm_path = vecs[-1] if vecs else None

    return model_path, vecnorm_path