import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_3d_plot():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "..", "data", "scaling", "S_scaling_summary.csv")
    if not os.path.exists(data_path):
        print(f"File {data_path} non trovato!")
        return
        
    n_list = []
    alpha_list = []
    tau_list = []
    
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = float(row["n_nodes"])
            a = float(row["alpha"])
            tau = float(row["tau_mean"])
            n_list.append(n)
            alpha_list.append(a)
            tau_list.append(tau)
            
    n_arr = np.array(n_list)
    a_arr = np.array(alpha_list)
    t_arr = np.array(tau_list)
    
    # Creiamo una griglia 2D regolare
    n_unique = sorted(list(set(n_arr)))
    a_unique = sorted(list(set(a_arr)))
    
    N, A = np.meshgrid(n_unique, a_unique)
    T = np.zeros_like(N, dtype=float)
    
    # Riempiamo la griglia
    for i in range(len(a_unique)):
        for j in range(len(n_unique)):
            # Troviamo il tau corrispondente
            match = t_arr[(n_arr == n_unique[j]) & (a_arr == a_unique[i])]
            if len(match) > 0:
                T[i, j] = match[0]
            else:
                T[i, j] = np.nan # Se mancano punti
                
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Disegniamo la superficie con gradiente "inferno"
    surf = ax.plot_surface(N, A, T, cmap='inferno_r', edgecolor='black', alpha=0.9)
    
    # Titoli
    ax.set_title("Orizzonte degli Eventi Topologico\nImpatto simultaneo di Rete (n) e Bias (α) sulla Vita (τ)", 
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Nodi della Rete (n)", fontsize=12, labelpad=10)
    ax.set_ylabel("Merchant Bias (α)", fontsize=12, labelpad=10)
    ax.set_zlabel("Vita Rete τ", fontsize=12, labelpad=10)
    
    # Invertiamo l'asse Z se vogliamo effetto 'buca' o coloriamo in base alla profondità
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Pressione di Fallimento (τ)')
    
    # Miglioriamo la telecamera
    ax.view_init(elev=30, azim=230)
    
    plots_dir = os.path.join(base_dir, "..", "..", "results", "img")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "3D_Alpha_Sink_Surface.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Grafico 3D Salvato: {out_path}")

if __name__ == "__main__":
    generate_3d_plot()
