# Proyecto asociado al paper "A Q-learning approach for long-term hydrothermal dispatch"

Este repositorio contiene el código, datos y resultados correspondientes al proyecto asociado al paper **"A Q-learning approach for long-term hydrothermal dispatch"**.  

---

## Estructura del repositorio

- **datos/**  
  Contiene todos los datos de entrada utilizados por los scripts (series históricas, matrices de transición hidrológica y energías determinísticas).

- **figures/**  
  Carpeta donde se generan automáticamente las figuras producidas al correr los scripts principales.  
  *Nota:* el script `visualizadorCronica.py` no guarda resultados en esta carpeta, ya que es una herramienta adicional de visualización.

- **resultados/**  
  Almacena los resultados obtenidos para los tres algoritmos probados:  
  - Actor-Critic  
  - PPO (Proximal Policy Optimization)  
  - Q-Learning  
  Cada uno evaluado bajo escenarios de aportes **estocásticos** y **determinísticos**.

- **salidas/**  
  Contiene archivos CSV con las distintas salidas de parámetros del problema generados en cada corrida.  
  *Nota:* el script `visualizadorCronica.py` **no** guarda salidas aquí.

- **A2C.py**  
  Script principal para definir, entrenar y evaluar el algoritmo **Actor-Critic**.

- **PPO.py**  
  Script principal para definir, entrenar y evaluar el algoritmo **PPO**.

- **Q_learning.py**  
  Script principal para definir, entrenar y evaluar el algoritmo **Q-Learning**.

- **visualizadorCronica.py**  
  Herramienta adicional que permite seleccionar uno de los CSV generados en `salidas/` y visualizar gráficamente (en forma apilada) las series de energía generada, demanda, volumen del embalse y aportes, todo en función de las semanas.

---

## Observaciones importantes

- Los scripts de **A2C** (`A2C.py`) y **PPO** (`PPO.py`) guardan en la **raíz del repositorio** el modelo entrenado cada vez que se ejecuta un entrenamiento.  
- Si se desea reentrenar alguno de estos algoritmos desde cero, es necesario **eliminar previamente el modelo guardado en la raíz**, para evitar que el script cargue automáticamente la versión anterior.

---

## Autores

- Florencia Roque 
- Matías Rama  
- Ignacio Salas 
- Mónica Carle
- Rodrigo Porteiro