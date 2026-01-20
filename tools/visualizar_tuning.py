# tools/visualizar_tuning.py

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_latest_tuning():
    # Busca el archivo CSV más reciente en la carpeta de tuning
    list_of_files = glob.glob('results/tuning/*.csv')
    if not list_of_files:
        print("No se encontraron archivos de tuning.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Visualizando: {latest_file}")
    
    df = pd.DataFrame(pd.read_csv(latest_file))
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['trial'], df['score'], marker='o', linestyle='-')
    plt.title(f'Progreso de Optuna - {os.path.basename(latest_file)}')
    plt.xlabel('Trial')
    plt.ylabel('Recompensa Media (Score)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_latest_tuning()