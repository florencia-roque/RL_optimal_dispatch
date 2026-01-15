# Proyecto asociado al paper "A Q-learning approach for long-term hydrothermal dispatch"

Este repositorio contiene el código, datos y resultados correspondientes al proyecto asociado al paper **"A Q-learning approach for long-term hydrothermal dispatch"**.  

---

## Estructura del repositorio

### TODO: ESCRIBIR ESTO, EXPLICANDO QUÉ ES CADA CARPETA Y ARCHIVO

## Configuración y Ejecución

1. **Clonar el repositorio:**
  ```bash
   git clone https://github.com/florencia-roque/RL_optimal_dispatch.git

   cd RL_optimal_dispatch
  ```

2. **Configuración del Entorno (opcional pero recomendado)**
Se recomienda el uso de un entorno virtual para evitar conflictos de dependencias. Los siguientes comandos ejecutarlos en una consola dentro de VS Code.

Para que permita la creacion de un entorno virtual, ejecutar primero:
  ```bash
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

A continuación ejecutar:

  ```bash
  python -m venv .venv
  ```

Aparecerá una ventana que pregunta si se desea seleccionar el entorno virtual creado para el espacio de trabajo actual.

![select_env](docs/select_env.png)

Clickeamos en Yes.

Y por último hay que ejecutar:

  ```bash
  .\.venv\Scripts\activate
  ```

3. **Instalar dependencias:**
Las mismas se instalarán desde el archivo que las tiene detalladas usando el siguiente comando:
  ```bash
  pip install -r requirements.txt
   ```    

4. **Dos maneras para ejecutar**

* Ejecutar con VS Code (recomendado):
  * Abre el proyecto en VS Code.
  * Presiona F5 para debug o Ctrl+F5 para Run sin debug.
  * Selecciona el algoritmo y los parámetros en el menú que aparece en la parte superior.

* Ejecutar desde Terminal: 

 Si no usa VS Code, puede ejecutar con los siguientes comandos desde la raíz del proyecto **(tener en cuenta que los comandos de configuración de entorno virtual fueron dados para VS Code, sin embargo puede utilizar la herramienta de su comodidad para ese fin)**.

*Entrenamiento:*  

 ```bash
python -m main --alg ql --mode train --total-episodes 3000
```   
*Evaluación:*  

 ```bash
python -m main --alg ql --mode eval --mode-eval historico
``` 

*Entrenamiento y evaluación:*
 ```bash
python -m main --alg ql --mode train_eval --total-episodes 3000 --mode-eval historico
```   

Convenciones actuales del proyecto
---------------------------------
- Entrenamiento: siempre MODO='markov'.
- Evaluación:
    * si el env es determinístico (DETERMINISTICO==1): se evalúa en la misma tira
      (no corresponde pedir modo por consola).
    * si el env es estocástico (DETERMINISTICO==0): se puede evaluar en 'markov'
      o 'historico'.

Wrappers del entorno.
---------------------------------
Regla práctica:
- Todo lo que sea una transformación *del entorno Gym* (obs/action/reward) va en src/environment/wrappers.py.
- Todo lo que sea utilitario tabular (encode/decode bins, etc.) va en
  src/environment/utils_tabular.py.


Tools extra.
---------------------------------
Actualmente en desuso esos archivos (26/12/2025). De momento se guardan para eventualmente reciclar código.

---

## Autores

- Florencia Roque 
- Matías Rama  
- Ignacio Salas 
- Mónica Carle
- Magdalena Irurtia
- Rodrigo Porteiro