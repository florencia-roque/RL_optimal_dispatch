import pandas as pd
import os
from src.utils import config
from src.utils.paths import mode_tag, timestamp

def moda_custom(series):
    """
    Calcula la moda. Si hay empate o múltiples modas, devuelve la primera.
    Si la serie está vacía o es todo null, devuelve None.
    """
    m = series.mode()
    if not m.empty:
        return m.iloc[0]
    return None

def promedio_redondeado(series):
    """
    Calcula el promedio y lo redondea al entero más cercano.
    Si la serie está vacía o es todo null, devuelve None.
    """
    if series.isnull().all() or series.empty:
        return None
    return 19 if int(series.mean().round()) == 20 else int(series.mean().round())

class AverageSeeds:
    def main(algoritmo):
    
        fecha_hora = timestamp()
        mode_tag_str = mode_tag(0, "markov",multiple_seeds=True)

        CARPETA_ORIGEN = config.EVALUATIONS / algoritmo
        CARPETA_DESTINO_MULTIPLE_SEEDS = CARPETA_ORIGEN / f"eval_{fecha_hora}_{mode_tag_str}_promedio_seeds"
        CARPETA_DESTINO_PROM_MULTIPLE_SEEDS = CARPETA_DESTINO_MULTIPLE_SEEDS / "promedios"

        # Crear carpeta donde irán los 114 promedios de las 20 evaluaciones con distinta semilla
        os.makedirs(CARPETA_DESTINO_MULTIPLE_SEEDS, exist_ok=True)
        # Crear carpeta donde irán los promedios de los promedios de las 114 evaluaciones con distinta semilla
        os.makedirs(CARPETA_DESTINO_PROM_MULTIPLE_SEEDS, exist_ok=True)

        # Definimos columnas discretas vs continuas
        if algoritmo in ["ppo", "a2c"]:
            COLUMNAS_DISCRETAS = ["hidrologia", "tiempo", "episode_id"]
            COLUMNAS_CONTINUAS = ["volumen", "volumen_turbinado", "energia_hidro", "energia_eolica",
                                   "energia_solar", "energia_biomasa", "energia_renovable", "energia_termico_bajo", "energia_termico_alto", "energia_exportada",
                                   "costo_termico", "ingreso_exportacion", "demanda", "demanda_residual", "fraccion_turbinado", "qt_max_fisico", "energia_hidro_max_frac",
                                   "energia_hidro_obj", "aportes", "vertimiento", "action", "reward"]
        elif algoritmo == "ql":
            COLUMNAS_DISCRETAS = ["volumen_discreto", "action", "hidrologia", "tiempo", "episode_id"] 
            COLUMNAS_CONTINUAS = ["volumen","volumen_turbinado", "energia_hidro", "energia_eolica",
                                   "energia_solar", "energia_biomasa", "energia_renovable", "energia_termico_bajo", "energia_termico_alto", "energia_exportada",
                                   "costo_termico", "ingreso_exportacion", "demanda", "demanda_residual", "fraccion_turbinado", "qt_max_fisico", "energia_hidro_max_frac",
                                   "energia_hidro_obj", "aportes", "vertimiento", "reward"]

        reglas = {}
        for col in COLUMNAS_CONTINUAS:
            reglas[col] = 'mean'
        for col in COLUMNAS_DISCRETAS:
            reglas[col] = promedio_redondeado

        print("Iniciando procesamiento de escenarios...")

        # Buscar dentro de todas las carpetas de evaluación con distinta semilla
        # Chequear si las carpetas finalizan en est_markov y guardarlas en una lista de subcarpetas
        subcarpetas = [
            CARPETA_ORIGEN / d for d in os.listdir(CARPETA_ORIGEN)
            if (CARPETA_ORIGEN / d).is_dir() and d.__contains__("est_markov")
        ]

        # Promediar los CSV de cada escenario entre todas las semillas 
        for esc in range(114):  # Suponiendo 114 escenarios
            dfs_seeds = []
           
            for subcarpeta in subcarpetas:
                if str(subcarpeta).endswith("promedio_seeds"):
                    pass
                else:
                    archivo = subcarpeta / f"escenario_{esc}.csv"
                    df = pd.read_csv(archivo)
                    df = df.reset_index(drop=True)
                    dfs_seeds.append(df) 
                    
            # La concatenación de los dataframes de las distintas semillas mismo escenario
            df_concat = pd.concat(dfs_seeds)
            
            # Promediar usando las reglas definidas y agrupando por índice (level=0)
            # Repite para el índice 1, 2... hasta el 155.
            df_resultado = df_concat.groupby(level=0).agg(reglas)
            
            # Guardar el CSV promedio en la carpeta de destino
            # Verificación de seguridad: El resultado debe tener 156 filas
            if len(df_resultado) != 156:
                print(f"OJO: El escenario {esc} resultó con {len(df_resultado)} filas en lugar de 156.")

            # Guardar archivo individual
            df_resultado.to_csv(os.path.join(CARPETA_DESTINO_MULTIPLE_SEEDS, f"escenario_{esc}.csv"), index=False)        
               
        # Promediar los CSV promedios entre todas las semillas
        nombres_archivos_promedios = ["costos.csv", "energias.csv", "estados.csv", "resultados_agente.csv", "trayectorias.csv"]
        dfs_seed_promedios = []

        for nombre_archivo in nombres_archivos_promedios:
            for subcarpeta in subcarpetas:
                if str(subcarpeta).endswith("promedio_seeds"):
                    pass
                else: 
                    df_promedios = pd.read_csv(subcarpeta / "promedios" / nombre_archivo)
                    df_promedios = df_promedios.reset_index(drop=True)
                    dfs_seed_promedios.append(df_promedios)

            df_concat_promedios = pd.concat(dfs_seed_promedios)

            # Filtrar el diccionario de reglas para que solo use columnas existentes
            reglas_validas = {k: v for k, v in reglas.items() if k in df_concat_promedios.columns}
            
            df_resultado_promedios = df_concat_promedios.groupby(level=0).agg(reglas_validas)
            df_resultado_promedios.to_csv(os.path.join(CARPETA_DESTINO_PROM_MULTIPLE_SEEDS, nombre_archivo), index=False)
       
        print("Procesamiento multi semilla completado.") 