# Reconfiguración gráfica y GPU para el proyecto RL Optimal Dispatch

## Objetivo

Documentar el proceso realizado para reconfigurar el entorno de ejecución del proyecto, incluyendo la habilitación de soporte de GPU para el entrenamiento del algoritmo PPO y la validación del funcionamiento del entorno.

## Fecha

15 de julio de 2026

## Contexto

El proyecto utiliza PyTorch junto con Stable-Baselines3 para entrenar agentes de aprendizaje por refuerzo. Para aprovechar la aceleración de hardware, se verificó si el entorno Python tenía soporte de CUDA disponible y, de no ser así, se procedió a instalar la versión de PyTorch compatible con la GPU presente en la máquina.

## Hardware detectado

- GPU: NVIDIA RTX A1000
- Driver NVIDIA instalado: 581.42
- Soporte CUDA detectado por `nvidia-smi`: CUDA 13.0

## Problema identificado

El entorno virtual del proyecto estaba usando una instalación de PyTorch sin soporte CUDA:

```powershell
2.9.1+cpu
False
0
```

Esto impedía que Stable-Baselines3/PyTorch ejecutaran operaciones sobre GPU y, por tanto, el entrenamiento de PPO se realizaba en CPU.

## Proceso realizado

### 1. Verificación del entorno Python

Se comprobó la versión de Python y la disponibilidad de PyTorch en el entorno virtual del proyecto:

```powershell
D:\RL_optimal_dispatch\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 2. Verificación del hardware NVIDIA

Se comprobó que la máquina tenía una GPU NVIDIA disponible:

```powershell
nvidia-smi
```

Resultado: la GPU RTX A1000 estaba visible y accesible por el sistema.

### 3. Instalación de PyTorch con soporte CUDA

Se instaló una versión de PyTorch compatible con CUDA 12.4 en el entorno virtual del proyecto:

```powershell
D:\RL_optimal_dispatch\.venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Verificación final

Se validó nuevamente que el entorno reconociera la GPU:

```powershell
D:\RL_optimal_dispatch\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Resultado verificado:

```text
2.6.0+cu124
cuda_available True
device_count 1
device_name NVIDIA RTX A1000
```

## Resultado final

El entorno quedó reconfigurado correctamente para usar GPU en el entrenamiento del algoritmo PPO.

El proyecto ya estaba preparado para usar GPU en el archivo de entrenamiento PPO, que utiliza:

```python
device="cuda" if torch.cuda.is_available() else "cpu"
```

Con la nueva instalación, el valor de `torch.cuda.is_available()` pasó de `False` a `True`, por lo que el entrenamiento podrá ejecutarse sobre GPU.

## Recomendaciones

- Mantener el entorno virtual del proyecto aislado del sistema Python global.
- Si se crea otro entorno virtual, repetir la instalación de PyTorch con CUDA para ese entorno.
- Verificar periódicamente que la instalación siga siendo compatible con la GPU y con la versión de Python utilizada.
- Para entrenar, ejecutar el proyecto desde el entorno virtual del proyecto y no desde otro intérprete.

## Comandos útiles

### Activar el entorno virtual

```powershell
D:\RL_optimal_dispatch\.venv\Scripts\Activate.ps1
```

### Verificar GPU en PyTorch

```powershell
D:\RL_optimal_dispatch\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### Ejecutar entrenamiento PPO

```powershell
D:\RL_optimal_dispatch\.venv\Scripts\python.exe main.py --alg ppo --mode train --total-episodes 8000 --det 0 --mode-eval historico
```
