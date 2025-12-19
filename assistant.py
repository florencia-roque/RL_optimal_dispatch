# assistant.py
"""
Interfaz interactiva para ejecutar PPO, A2C o Q-learning.
Permite entrenar o evaluar modelos guardados.
"""

from __future__ import annotations
from pathlib import Path

from src.rl_algorithms.ppo_agent import PPOAgent
from src.rl_algorithms.a2c_agent import A2CAgent
from src.rl_algorithms.q_learning_agent import QLearningAgent

from src.utils.paths import get_latest_model
from src.utils.metrics import compute_all_metrics


# ============================================================
# FUNCIONES DE MENÚ
# ============================================================

def menu_select(prompt, options):
    print("\n" + prompt)
    for i, opt in enumerate(options, start=1):
        print(f"{i}. {opt}")
    while True:
        choice = input("Elegí opción: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Opción inválida. Intentá de nuevo.")


def list_models(alg: str):
    """Devuelve lista de modelos guardados en models/<alg> (si existe esa convención)."""
    models_dir = Path("models") / alg
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.zip"))


# ============================================================
# PRINCIPAL
# ============================================================

def main():
    print("\n========== DESPACHO HIDRO-TÉRMICO - INTERFAZ RL ==========\n")

    # 1) Elegir algoritmo
    alg = menu_select("¿Qué algoritmo querés usar?", ["ppo", "a2c", "ql"])

    # Crear agente
    if alg == "ppo":
        agent = PPOAgent()
    elif alg == "a2c":
        agent = A2CAgent()
    else:
        agent = QLearningAgent()

    # 2) Elegir acción
    accion = menu_select(f"¿Qué querés hacer con {alg.upper()}?", ["Entrenar", "Evaluar modelo guardado"])

    # 3) Entrenamiento
    if accion == "Entrenar":
        print("\nEntrenando modelo...")

        episodes = 3000 if alg == "ql" else 2000
        agent.train(total_episodes=episodes)

        print("\n✔ Entrenamiento finalizado. El modelo fue guardado automáticamente.\n")
        return

    # 4) Evaluación
    print(f"\n=== Evaluación de {alg.upper()} ===")

    modelos = list_models(alg)
    if len(modelos) == 0:
        print("\n❌ No hay modelos guardados para este algoritmo en models/<alg>.")
        print("Tip: si ahora guardás en otra estructura (training_paths), actualizamos list_models/get_latest_model.")
        return

    choice = menu_select(
        "¿Qué modelo querés evaluar?",
        ["Último modelo guardado"] + [str(m.name) for m in modelos]
    )

    if choice == "Último modelo guardado":
        out = get_latest_model(alg)
        # por compatibilidad si devuelve 1 o 2 cosas
        if isinstance(out, (tuple, list)) and len(out) == 2:
            model_path, vecnorm_path = out
        else:
            model_path, vecnorm_path = out, None
    else:
        model_path = Path("models") / alg / choice
        vecnorm_path = None
        if alg == "ppo":
            vecnorms = sorted((Path("models") / alg).glob("vecnorm*.pkl"))
            vecnorm_path = vecnorms[-1] if vecnorms else None

    print(f"\nCargando modelo: {model_path.name}")

    # 5) Elegir modo de evaluación
    modo_eval = menu_select(
        "Elegí el tipo de evaluación:",
        ["markov", "historico"]
        # Si querés volver a exponer determinístico cuando el factory lo soporte:
        # ["markov", "historico", "deterministico"]
    )

    # 6) Cargar modelo con modo_eval (importante para que el env se cree bien)
    if alg == "ppo":
        agent.load(model_path, vecnorm_path, modo_eval=modo_eval)
    elif alg == "a2c":
        agent.load(model_path, modo_eval=modo_eval)
    else:
        # QL: hoy tu load() no recibe modo_eval; si lo querés, lo agregamos.
        agent.load(model_path)

    # 7) Ejecutar evaluación
    print("\nEvaluando...")

    if alg in ("ppo", "a2c"):
        # Sliding window paralelo
        df_avg, df_all = agent.evaluate(
            n_eval_episodes=114,
            window_weeks=156,
            stride_weeks=52,
            n_envs=8,
            modo_eval=modo_eval,
        )
    else:
        # Q-learning tabular (por ahora sin sliding window paralelo)
        df_avg, df_all = agent.evaluate(
            n_eval_episodes=114,
            num_pasos=155,
        )

    # 8) Métricas
    metrics = compute_all_metrics(df_avg, df_all)

    print("\n===== MÉTRICAS DEL MODELO =====\n")
    for k, v in metrics.items():
        print(f"{k:25s} : {v}")

    print("\n✔ Evaluación finalizada. Resultados guardados en /evaluations.\n")


if __name__ == "__main__":
    main()
