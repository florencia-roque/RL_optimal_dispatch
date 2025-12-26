# src/utils/config.py

"""
Define carpetas base y rutas de datos usadas en el proyecto.
"""
from pathlib import Path

# Raíz del repo
ROOT = Path(__file__).resolve().parents[2]

# Datos
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Datos de Claire
CLAIRE_APORTES_HIST_XLSX = DATA_RAW / "markov" / "aporte_historico_claire.xlsx" # fue estimado con MOP
CLAIRE_XLT = DATA_RAW / "claire" / "datosProcHistorico.xlt"
CLAIRE_APORTE_CSV = DATA_PROCESSED / "aporte_claire.csv"
CLAIRE_HIDROLOGIA_CSV = DATA_PROCESSED / "hidrologia_claire.csv"
CLAIRE_MATRICES_CSV = DATA_PROCESSED / "matrices_markov_claire.csv"

# MOP determinísticos (energias renovables y demanda)
MOP_DET_XLSX = DATA_RAW / "mop" / "energias_ernc_demanda.xlsx"   # biomasa/eólico/solar/demanda
MOP_APORTES_DET_XLSX = DATA_RAW / "mop" / "aportes_deterministicos.xlsx"  # para entrenar con aportes deterministicos

# Resultados
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS  = RESULTS / "models"
EVALUATIONS = RESULTS / "evaluations"

# Crear carpetas si no existen
for p in [
    DATA_PROCESSED,
    CLAIRE_APORTE_CSV.parent,
    CLAIRE_HIDROLOGIA_CSV.parent,
    CLAIRE_MATRICES_CSV.parent,
    FIGURES,
    MODELS,
]:
    p.mkdir(parents=True, exist_ok=True)