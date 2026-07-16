# RL_optimal_dispatch

Reinforcement Learning framework for **long-term hydrothermal dispatch optimization**.  

It provides a reproducible pipeline to train and evaluate reinforcement learning (RL) agents for long-term hydrothermal dispatch problems under stochastic (Markov) hydrology. It also gives the possibility to evaluate under historical hydrological and deterministic inflows. 

---

## Overview

The project implements and compares different RL algorithms (Q-learning, PPO, A2C) on a hydrothermal dispatch environment that models reservoir dynamics, thermal generation costs, renewable generation and demand uncertainty.

The codebase is organized to clearly separate:
- environment definition
- RL algorithms
- preprocessing of hydrological data
- evaluation and visualization
- experiment configuration and utilities

---

## Requirements

- Python **3.10** or **3.11**
- pip
- Git (optional)

> Python >= 3.12 may cause incompatibilities with Stable-Baselines3.

---

## Obtaining the project
**If git is used**, the following commands can prepare the workspace.

  ```bash
   git clone https://github.com/florencia-roque/RL_optimal_dispatch.git

   cd RL_optimal_dispatch
  ```

## Installation

All commands must be executed from the **project root directory** (where `main.py` is located).

### Windows (PowerShell)

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
```bash
python -m venv .venv
```
```bash
.\.venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
### Linux / macOS

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

If activated, the environment name will appear before the path, like this: 
```bash
(venv) path/to/project.
```

When need to run the code and the environment is not activated, the following command needs to be executed:

  ```bash
  .\.venv\Scripts\activate
  ```

## Entry point
```bash
main.py
```
Main entry point of the project.

This script is responsible for:
* parsing command-line arguments
* selecting the RL algorithm
* creating the appropriate environment
* launching training and/or evaluation
* saving trained models and evaluation outputs

All experiments (training, evaluation, tuning) are executed through this file.

## Project structure
RL_optimal_dispatch/

│── main.py

│── requirements.txt

│── README.md

├── data/

├── src/

├── results/

└── tools/

## Dependencies
```bash
requirements.txt
```

Lists all Python dependencies required to run the project, including:

* numerical libraries (NumPy, pandas)
* plotting libraries (matplotlib)
* Gymnasium
* Stable-Baselines3
* auxiliary scientific and utility packages

## Data
```bash
data/raw/
```
Contains raw input datasets used by the project.
These files are not directly consumed by the environments and may require preprocessing.

* Historical hydrological inflows

* Markov-chain related input files

* Demand and renewable generation data

* Original spreadsheets from external models (e.g. MOP)

*The file /data/raw/claire/datosProcHistorico.xlt is used to obtain the sum of all water reservoirs, that models Claire.*

```bash
data/processed/
```
Contains processed datasets generated from data/raw/ and directly used by the environments.

Key files:

* aporte_claire.csv - processed inflow series for the aggregated reservoir
* hidrologia_claire.csv - hydrological state classification
* matrices_markov_claire.csv - Markov transition probability matrices

## Results

```bash
results/evaluations/
```
Stores RL models evaluations and checkpoints, organized by algorithm and experiment. In each folder of evaluation there are multiple csv files, one for each eval episode and there is another folder that contains the mean of all of these evaluations.

```bash
results/figures/
```
Stores figures generated during evaluation and analysis (training trajectories, csv with training data, chronicles dispatch evaluation).

```bash
results/logs/
```
Stores logs of the ppo algorithm.

```bash
results/models/
```
Stores trained RL models and checkpoints, organized by algorithm and experiment.

```bash
results/tuning/
```
Stores csv files containing the results of the different trials made with Optuna tuning.
## Execution

* VS Code execution (recommmended):
  * Open the project in VS Code.
  * Press F5 button to debug or Ctr+F5 to run without debug.
  * Select the algorithm and the other parameters from the menu on the top of the window.

* Command line execution:
  * If VS Code is not available for this purpose, the following commands can be executed in the root of the project.

*Training:*  

 ```bash
python -m main --alg ql --mode train --total-episodes 3000 --mode-eval historico
```   
*Evaluation:*  

 ```bash
python -m main --alg ql --mode eval --total-episodes 3000 --mode-eval historico
``` 

*Training and Evaluation:*
 ```bash
python -m main --alg ql --mode train_eval --total-episodes 3000 --mode-eval historico
```      

*Evaluation Multiple Seeds:*
```bash
python -m main --alg ql --mode eval_multiple_seeds --total-episodes 3000 --mode-eval historico
```

*Evaluation Multiple Seeds Posteval:*
```bash
python -m main --alg ql --mode eval_multiple_seeds_posteval --total-episodes 3000 --mode-eval historico
```

## Source code (src/)
### src/environment/

Defines Gymnasium-compatible environments and environment-related utilities.

```bash
env_factory.py
```

Factory module that builds and returns the correct environment instance depending on:

* RL algorithm
* tabular vs continuous formulation
* deterministic vs stochastic setting

```bash
hydrothermal_env_tabular.py
```

Tabular hydrothermal environment with discretized state and action spaces. *Used by Q-learning.*

```bash
hydrothermal_env_continuous.py
```

Continuous or SB3-compatible environment *used by PPO and A2C agents*.

```bash
wrappers.py
```

Gym wrappers that modify observations, actions or rewards without altering the core environment logic.

```bash
utils_tabular.py
```

Helper functions for tabular environments, including:

* discretization of continuous variables
* encoding and decoding of states and actions
* finding the optimal policy

#### Development rule for wrappers vs utils:

* Any transformation of the Gym environment (observation/action/reward) goes in src/environment/wrappers.py.
* Any utility specific to the tabular approach (bin encoding/decoding, etc.) goes in src/environment/utils_tabular.py.


### src/rl_algorithms/
Implements the reinforcement learning agents.

```bash
q_learning_agent.py
``` 

Complete implementation of tabular Q-learning:

* training loop
* epsilon-greedy exploration
* Q-table updates
* model saving and loading

```bash
ppo_agent.py
```

PPO agent wrapper built on top of Stable-Baselines3.

```bash
a2c_agent.py
```

A2C agent wrapper built on top of Stable-Baselines3.

### src/evaluation/
Handles evaluation, post-processing and result storage.

```bash
evaluator_sb3.py
```

Evaluation pipeline for SB3-based agents, including rollout execution and trajectory collection.

```bash
eval_config.py
```

Centralized configuration for evaluation parameters (number of episodes, scenarios, seeds).

```bash
eval_outputs.py
```

Utilities for saving evaluation results, including dataframes, summaries and serialized outputs.

### src/preprocessing/
Hydrological data preprocessing.

```bash
claire_inflows.py
```

Processes historical inflows and constructs the datasets required for the Markov hydrology representation.

### src/utils/
General-purpose utilities shared across the project.

```bash
config.py
```

Global configuration file containing constants and base paths.

```bash
paths.py
```

Helper functions to manage filesystem paths and locate saved models.

```bash
callbacks.py
```

Training callbacks (logging, progress tracking, visualization).

```bash
metrics.py
```

Metric definitions and helper functions for performance evaluation.

```bash
io.py
```

Input/output utilities for reading and writing experiment data.

```bash
hparam_tuning.py
```

Hyperparameter tuning utilities.

```bash
average_seeds.py
```

Runs multiple experiments with different random seeds and aggregates results.

## Tools
### tools/
Standalone scripts mainly used for visualization and analysis.

```bash
plot_chronicle.py
```

Visualization of historical trajectories and scenarios.

```bash
plot_tuning.py
```

Visualization of hyperparameter tuning results.

## extra
Deprecated or experimental scripts kept for reference.

## Repository configuration

```bash
.github/CODEOWNERS
```

Defines who is allowed to review and approve pull requests.
This file follows the standard GitHub CODEOWNERS mechanism and is used to automatically request reviews from designated maintainers when relevant parts of the repository are modified.

## Project conventions

The following conventions are currently adopted across experiments to ensure consistency:

**Training**

Training is always performed using MODO='markov'.

**Evaluation**

*If the environment is deterministic (DETERMINISTICO == 1):*

* Evaluation is performed on the same deterministic inflow trajectory.
* No evaluation mode needs to be specified via command line.

*If the environment is stochastic (DETERMINISTICO == 0):*

Evaluation can be performed either in:
* markov mode (in-sample evaluation), or
* historico mode (out-of-sample evaluation using historical chronicles).

## Reproducibility

* Trained models are saved under results/models/.
* Results are reproducible given identical: seed, dataset, trained model.

## Versioning notes
The following files and folders should not be committed:

* .venv/
* \_\_pycache\_\_/
* large results/ folders
* .git/ when sharing the project as a ZIP archive

## License and usage
This project is intended for academic and research purposes only.

## Authors
* Florencia Roque
* Matías Rama
* Ignacio Salas
* Mónica Carle
* Magdalena Irurtia
* Rodrigo Porteiro