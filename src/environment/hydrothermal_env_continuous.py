# src/environment/hydrothermal_env_continuous.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.utils.config import (
    MOP_DET_XLSX,
    MOP_APORTES_DET_XLSX,
    CLAIRE_APORTES_HIST_XLSX,
    CLAIRE_APORTE_CSV,
    CLAIRE_HIDROLOGIA_CSV,
    CLAIRE_MATRICES_CSV
)

from src.utils.io import leer_archivo

class HydroThermalEnvCont(gym.Env):
    T0 = 0
    T_MAX = 155
    N_HIDRO = 5

    P_CLAIRE_MAX = 1541 # MW
    P_SOLAR_MAX = 254 # MW
    P_EOLICO_MAX = 1584.7 # MW
    P_BIOMASA_MAX = 487.3 # MW
    P_TERMICO_BAJO_MAX = 1300 # MW
    P_TERMICO_ALTO_MAX = 5000 # MW

    Q_CLAIRE_MAX = 11280 * 3600 / 1e6 # hm3/h

    V_CLAIRE_MIN = 0 # hm3
    V_CLAIRE_MAX = 60000 # hm3
    V0 = V_CLAIRE_MAX * 0.75 # hm3
    
    K_CLAIRE = P_CLAIRE_MAX / Q_CLAIRE_MAX # MWh/hm3

    V_CLAIRE_TUR_MAX = P_CLAIRE_MAX * 168 / K_CLAIRE # hm3

    VALOR_EXPORTACION = 1e-6 # USD/MWh 
    COSTO_TERMICO_BAJO = 100.0 # USD/MWh 
    COSTO_TERMICO_ALTO = 300.0 # USD/MWh
    COSTO_VERTIMIENTO = 0.0 # USD/MWh

    # poner 0 si queremos usar aportes estocásticos
    DETERMINISTICO = 0

    MODO = "markov"

    def __init__(self):

        # Espacio de observación
        self.observation_space = spaces.Dict({
            "volumen": spaces.Box(self.V_CLAIRE_MIN, self.V_CLAIRE_MAX, shape=(), dtype=np.float32),
            "hidrologia": spaces.Discrete(self.N_HIDRO, start=0),
            "tiempo": spaces.Discrete(self.T_MAX + 1, start=0)
        })
        
        # Espacio de acción continuo: fracción de turbinado en [0, 1]
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        
        # Cargar datos determinísticos de MOP (energias renovables y demanda)
        self.data_biomasa = leer_archivo(str(MOP_DET_XLSX), header=0, sheet_name=0).iloc[:, 1:]
        self.data_eolico  = leer_archivo(str(MOP_DET_XLSX), header=0, sheet_name=1).iloc[:, 1:]
        self.data_solar   = leer_archivo(str(MOP_DET_XLSX), header=0, sheet_name=2).iloc[:, 1:]
        self.data_demanda = leer_archivo(str(MOP_DET_XLSX), header=0, sheet_name=3).iloc[:, 1:]

        # Agregar columna con promedio de crónicas
        self.data_biomasa["PROMEDIO"] = self.data_biomasa.mean(axis=1)
        self.data_eolico["PROMEDIO"] = self.data_eolico.mean(axis=1)
        self.data_solar["PROMEDIO"] = self.data_solar.mean(axis=1)
        self.data_demanda["PROMEDIO"] = self.data_demanda.mean(axis=1)

        # Cargar matriz de aportes discretizada (con estado hidrológico 0,1,2,3,4)
        self.data_matriz_aportes_discreta = leer_archivo(str(CLAIRE_HIDROLOGIA_CSV), sep=",", header=0)
        self.aportes_deterministicos = leer_archivo(str(MOP_APORTES_DET_XLSX), sep=",", header=0)
        
        # Cargar matriz de aportes continuos (unidad de los aportes de Claire: m3/s )
        self.data_matriz_aportes_claire = leer_archivo(str(CLAIRE_APORTE_CSV), sep=",", header=0)

        # Convertir a unidad hm3/h
        self.data_matriz_aportes_claire = self.data_matriz_aportes_claire * 3600 / 1e6

        # Cargar datos de matrices hidrológicas con las probabilidades de transición entre estados
        self.data_matrices_hidrologicas = leer_archivo(str(CLAIRE_MATRICES_CSV), sep=",", header=0).iloc[:, 1:]

        self.matrices_hidrologicas = {}
        for i in range(self.data_matrices_hidrologicas.shape[0]):
            array_1d = self.data_matrices_hidrologicas.iloc[i, :].values
            self.matrices_hidrologicas[i] = array_1d.reshape(5, 5) 

        # Leer archivo de aportes historicos
        self.datos_historicos = leer_archivo(str(CLAIRE_APORTES_HIST_XLSX), header=7)

        self.indice_inicial_episodio = 0
        self.episodios_recorridos = 0

    def reset(self, seed=None, options=None):
        # IMPORTANTE: inicializa el RNG del entorno
        super().reset(seed=seed)

        if options and "start_week" in options:
            self.indice_inicial_episodio = int(options["start_week"])
            # self.episodios_recorridos = self.indice_inicial_episodio // 52
        else:
            self.indice_inicial_episodio = self.episodios_recorridos * 52
            self.episodios_recorridos += 1
        
        if self.DETERMINISTICO == 0 and self.MODO not in {"markov", "historico"}:
            raise ValueError("modo debe ser 'markov' u 'historico'")

        self.volumen = self.V0
        self.tiempo = 0
        self.vertimiento = 0.0
        self.hidrologia = self._inicial_hidrologia()
        self.hidrologia_anterior = self.hidrologia

        info = {
            "volumen_inicial": self.volumen,
            "hidrologia_inicial": self.hidrologia,
            "tiempo_inicial": self.tiempo
        }

        return self._get_obs(), info
    
    def _inicial_hidrologia(self):
        if self.DETERMINISTICO == 1:
            return int(self.aportes_deterministicos.iloc[self.tiempo, 1])
        elif self.MODO == "markov":
            return np.int64(2)
        else:
            return int(self.datos_historicos.iloc[self.indice_inicial_episodio, 3])
        
    def _siguiente_hidrologia(self):
        self.hidrologia_anterior = self.hidrologia
        if self.DETERMINISTICO == 1:
            return int(self.aportes_deterministicos.iloc[self.tiempo + 1, 1])
        elif self.MODO == "markov":
            # retorna el estado hidrológico siguiente 0,1,2,3,4
            clases = np.arange(self.matrices_hidrologicas[self.tiempo % 52].shape[0])
            # USAR el RNG del env (no el global):
            return self.np_random.choice(clases,p=self.matrices_hidrologicas[self.tiempo % 52][self.hidrologia,:])
        else:
            return int(self.datos_historicos.iloc[self.indice_inicial_episodio + self.tiempo, 3])
    
    def _aporte(self):
        # MODO MARKOV PUEDE USARSE PARA ENTRENAR O EVALUAR
        if self.MODO == "markov" and self.DETERMINISTICO == 0:
            # Guardo fila de estados para la semana actual
            estados_t = self.data_matriz_aportes_discreta.loc[self.tiempo % 52] 

            # Guardo las columnas que tienen el eshy actual
            coincidencias = (estados_t == self.hidrologia)
            cronicas_coincidentes = coincidencias[coincidencias].index

            # Con las cronicas coincidentes tengo que obtener los aportes para la semana y eshy actual
            aportes = self.data_matriz_aportes_claire.loc[self.tiempo % 52, cronicas_coincidentes] # hm3/h

            # Calculo la media de los aportes para la semana y eshy actual
            aportes_promedio = np.mean(aportes) # hm3/h

            rango_valido_inf = aportes_promedio - aportes_promedio * 0.05
            rango_valido_sup = aportes_promedio + aportes_promedio * 0.05

            # Me quedo con los aportes que estén en el promedio +/- 5% 
            aportes_validos = aportes[(aportes>=rango_valido_inf) & (aportes<=rango_valido_sup)] # hm3/h

            # Si aportes_validos es vacio tomo como aporte valido el promedio de aportes
            if aportes_validos.empty:
                aporte_final = aportes_promedio * 168   # hm3/semana
            else:
            # Sorteo uniformemente uno de los validos
                aporte_final = self.np_random.choice(aportes_validos) * 168 
            
            return aporte_final 

        # SOLO PARA EVALUAR CON LA HISTORIA HABIENDO ENTRENADO CON MARKOV 
        # NO SE ENTRENA CON LA HISTORIA 
        elif self.MODO == "historico" and self.DETERMINISTICO == 0:
            return self.datos_historicos.iloc[self.indice_inicial_episodio + self.tiempo, 2] * 3600 * 168 / 1e6 # hm3/semana
        
        # SIRVE PARA ENTRENAR Y EVALUAR CON LA TIRA DE APORTES DETERMINISTICOS
        elif self.DETERMINISTICO == 1:    
            valor = self.aportes_deterministicos.iloc[self.tiempo, 0] * 3600 * 168 / 1e6 # hm3/semana
            return valor

    def _demanda(self):
        # Obtener demanda de energía para el tiempo actual
        energias_demandas = self.data_demanda["PROMEDIO"]
        if self.tiempo < len(energias_demandas):
            # ESTO ESTA COMENTADO PORQUE AHORA SE AUMENTO EN EL MOP A POR 1.2 ENTONCES LA DEMANDA YA VIENE MAS GRANDE, HAY QUE USARLA ASI COMO VIENE (SIN MULTIPLICAR)
            # return energias_demandas.iloc[self.tiempo] * 1.2 
            return energias_demandas.iloc[self.tiempo]
        else:
            raise ValueError("Tiempo fuera de rango para datos de demanda")
    
    def _gen_eolico(self):
        # Obtener generación eólica para el tiempo actual
        energias_eolico = self.data_eolico["PROMEDIO"]
        if self.tiempo < len(energias_eolico):
            return 0
        else:
            raise ValueError("Tiempo fuera de rango para datos eólicos")

    def _gen_solar(self):
        # Obtener generacion solar para el tiempo actual 
        energias_solar = self.data_solar["PROMEDIO"]
        if self.tiempo < len(energias_solar):
            return 0
        else:
            raise ValueError("Tiempo fuera de rango para datos solares")

    def _gen_bio(self):
        # Obtener generacion de biomasa para el tiempo actual
        energias_biomasa = self.data_biomasa["PROMEDIO"]
        if self.tiempo < len(energias_biomasa):
            return 0
        else:
            raise ValueError("Tiempo fuera de rango para datos biomasa")

    def _gen_renovable(self):
        # Generacion total de energias renovables no convencionales
        return self._gen_eolico() + self._gen_solar() + self._gen_bio()

    def _gen_termico_bajo(self, demanda_residual):
        if demanda_residual <= self.P_TERMICO_BAJO_MAX * 168:
            return demanda_residual
        else:
            return self.P_TERMICO_BAJO_MAX * 168

    def _gen_termico_alto(self, demanda_residual):
        if demanda_residual <= self.P_TERMICO_ALTO_MAX * 168:
            return demanda_residual
        else:
            return self.P_TERMICO_ALTO_MAX * 168
        
    def _despachar(self, qt):
        # Demanda residual
        demanda_residual = self._demanda() - self._gen_renovable() # MWh

        # Energia hidro
        energia_hidro = self.K_CLAIRE * qt # MWh
        demanda_residual -= energia_hidro
        
        # Termico bajo
        energia_termico_bajo = self._gen_termico_bajo(max(demanda_residual, 0.0)) # MWh
        demanda_residual -= energia_termico_bajo
        
        # Termico alto
        energia_termico_alto = self._gen_termico_alto(max(demanda_residual, 0.0)) # MWh
        demanda_residual -= energia_termico_alto

        # Energia exportada
        energia_exportada = np.abs(demanda_residual) # MWh

        # Ingreso por exportacion
        ingreso_exportacion = energia_exportada * self.VALOR_EXPORTACION # USD

        # Costo de energia termica
        costo_termico = (energia_termico_bajo * self.COSTO_TERMICO_BAJO + 
                         energia_termico_alto * self.COSTO_TERMICO_ALTO) # USD
        
        return ingreso_exportacion, costo_termico, energia_exportada, energia_termico_bajo, energia_termico_alto, energia_hidro

    def step(self, action):
        # Validar que la accion este en el espacio valido
        action = np.array(action, dtype=np.float32).reshape(1,)
        assert self.action_space.contains(action), f"Acción inválida: {action}. Debe estar en {self.action_space}"
        frac = float(action[0])

        # Demanda residual considerando solo renovables (sin hidro)
        demanda_residual_pre = max(self._demanda() - self._gen_renovable(), 0.0)  # MWh

        # Maximo fisico semanal (por potencia y por volumen) en hm3
        qt_max_fisico = min(frac * self.V_CLAIRE_TUR_MAX, self.volumen)

        # Limita la hidro a no exceder la demanda residual (evita exportar con hidro)
        energia_hidro_max_frac = self.K_CLAIRE * qt_max_fisico  # MWh
        energia_hidro_obj = min(energia_hidro_max_frac, demanda_residual_pre)  # MWh
        qt = energia_hidro_obj / self.K_CLAIRE  # hm3

        # Despacho: e_eolo + e_sol + e_bio + e_termico + e_hidro = dem + exp
        ingreso_exportacion, costo_termico, energia_exportada, energia_termico_bajo, energia_termico_alto, energia_hidro = self._despachar(qt)

        info = {
            "volumen": self.volumen,
            "hidrologia": self.hidrologia,
            "tiempo": self.tiempo,
            "volumen_turbinado": qt,
            "energia_hidro": energia_hidro,
            "energia_eolica": self._gen_eolico(),
            "energia_solar": self._gen_solar(),
            "energia_biomasa": self._gen_bio(),
            "energia_renovable": self._gen_renovable(),
            "energia_termico_bajo": energia_termico_bajo,
            "energia_termico_alto": energia_termico_alto,
            "energia_exportada": energia_exportada,
            "costo_termico": costo_termico,
            "ingreso_exportacion": ingreso_exportacion,
            "demanda": self._demanda(),
            "demanda_residual": self._demanda() - self._gen_renovable(),
            "fraccion_turbinado": frac,

            # Metricas de diagnóstico
            "qt_max_fisico": qt_max_fisico,
            "energia_hidro_max_frac": energia_hidro_max_frac,
            "energia_hidro_obj": energia_hidro_obj,
        }
        
        # Actualizar variables internas
        aporte_paso = self._aporte() # hm3 de la semana (volumen)
        self.hidrologia = self._siguiente_hidrologia()
        v_intermedio = self.volumen - qt + aporte_paso
        self.vertimiento = max(v_intermedio - self.V_CLAIRE_MAX, 0) 
        self.volumen = min(v_intermedio, self.V_CLAIRE_MAX) # hm3
        self.tiempo += 1
        
        info["aportes"] = aporte_paso
        info["vertimiento"] = self.vertimiento

        # Costo por vertimiento
        energia_vertida = self.vertimiento * self.K_CLAIRE
        costo_vertimiento = energia_vertida * self.COSTO_VERTIMIENTO # USD

        # Recompensa
        reward_usd = - costo_termico - costo_vertimiento + ingreso_exportacion  # USD
        reward = reward_usd / 1e6  # MUSD

        terminated = self.tiempo >= self.T_MAX
        truncated = False
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Semana {self.tiempo}:")
            print(f"  Volumen embalse: {self.volumen:.2f}/{self.V_CLAIRE_MAX}")
            print(f"  Estado hidrológico: {self.hidrologia}")
            print(f"  Porcentaje llenado: {(self.volumen/self.V_CLAIRE_MAX)*100:.1f}%")
            print("-" * 30)
        elif mode == 'rgb_array':
            # Retornar una imagen como array numpy para grabacion
            pass
        elif mode == 'ansi':
            # Retornar string para mostrar en terminal
            return f"T:{self.tiempo} V:{self.volumen:.1f} H:{self.hidrologia}"
        
    def _get_obs(self):
        # Mapeo de variables internas a observacion del agente
        obs = {
            "volumen": np.array(self.volumen, dtype=np.float32),
            "hidrologia": int(self.hidrologia),
            "tiempo": int(self.tiempo)
        }
        
        assert self.observation_space.contains(obs), f"Observación inválida: {obs}. Debe estar en {self.observation_space}"
        return obs