# tools/visualizar_tuning.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Si no tienes seaborn: pip install seaborn
import glob
import os

def plot_latest_tuning():
    # Busca el archivo CSV más reciente
    list_of_files = glob.glob('results/tuning/*.csv')
    if not list_of_files:
        print("No se encontraron archivos de tuning en results/tuning/")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Visualizando archivo: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Ordenamos por trial para que la línea salga bien
    df = df.sort_values("trial")
    
    # --- GRÁFICO 1: HISTORIA DE OPTIMIZACIÓN ---
    plt.figure(figsize=(10, 6))
    
    # Línea de progreso
    sns.lineplot(data=df, x='trial', y='score', marker='o', label='Trial')
    
    # Resaltar el mejor trial
    best_idx = df['score'].idxmax()
    best_trial = df.loc[best_idx]
    
    plt.scatter(best_trial['trial'], best_trial['score'], 
                color='red', s=150, zorder=5, label=f'Mejor ({best_trial["score"]:.2f})')
    
    plt.title(f'Historia de Optimización - {os.path.basename(latest_file)}')
    plt.xlabel('Número de Trial')
    plt.ylabel('Recompensa (Score)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # --- GRÁFICO 2: IMPACTO DE HIPERPARÁMETROS ---
    # Identificar columnas que son parámetros (excluyendo trial y score)
    param_cols = [c for c in df.columns if c not in ['trial', 'score']]
    
    if param_cols:
        n_params = len(param_cols)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5), sharey=True)
        
        if n_params == 1: axes = [axes]  # Asegurar que axes es iterable

        for i, param in enumerate(param_cols):
            # Scatter plot: x=parametro, y=score
            sns.scatterplot(data=df, x=param, y='score', ax=axes[i], 
                            hue='score', palette='viridis', s=100)
            
            axes[i].set_title(f'Impacto de {param}')
            axes[i].set_ylabel('Score')
            axes[i].grid(True, alpha=0.3)
            
            # Escala logarítmica si los datos varían en órdenes de magnitud
            if df[param].max() / (df[param].min() + 1e-9) > 100:
                axes[i].set_xscale('log')

        plt.suptitle(f'Análisis de Hiperparámetros')
        plt.tight_layout()
        plt.show()

    # --- REPORTE EN CONSOLA ---
    print("\n" + "="*40)
    print(f" RESULTADOS DEL TUNING")
    print("="*40)
    print(f"Mejor Recompensa: {best_trial['score']:.4f}")
    print(f"Encontrado en Trial: {int(best_trial['trial'])}")
    print("-" * 20)
    print("MEJORES PARÁMETROS:")
    for param in param_cols:
        print(f"  * {param}: {best_trial[param]}")
    print("="*40 + "\n")

if __name__ == "__main__":
    plot_latest_tuning()