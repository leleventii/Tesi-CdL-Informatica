import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_3d_nk_plot():
    data_path = "data_multi/B_tau_vs_k_summary.csv"
    if not os.path.exists(data_path):
        print(f"File {data_path} non trovato!")
        return
        
    n_list = []
    k_list = []
    tau_list = []
    
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Controlliamo che stiamo leggendo l'exp sui k e che alpha sia 0.5
                alpha_val = float(row.get("alpha", 0.0))
                if alpha_val != 0.5:
                    continue
                
                n = float(row["n_nodes"])
                k_val = float(row["param_value"])
                tau = float(row["tau_mean"])
                
                n_list.append(n)
                k_list.append(k_val)
                tau_list.append(tau)
            except Exception as e:
                pass
            
    if not n_list:
         print("Nessun dato con alpha=0.5 trovato.")
         return

    n_arr = np.array(n_list)
    k_arr = np.array(k_list)
    t_arr = np.array(tau_list)
    
    n_unique = sorted(list(set(n_arr)))
    k_unique = sorted(list(set(k_arr)))
    
    N, K = np.meshgrid(n_unique, k_unique)
    T = np.zeros_like(N, dtype=float)
    
    for i in range(len(k_unique)):
        for j in range(len(n_unique)):
            match = t_arr[(n_arr == n_unique[j]) & (k_arr == k_unique[i])]
            if len(match) > 0:
                T[i, j] = match[0]
            else:
                T[i, j] = np.nan
                
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Cambiamo colore in viridis per questo k,tau
    surf = ax.plot_surface(N, K, T, cmap='viridis', edgecolor='black', alpha=0.9)
    
    ax.set_title("Analisi Parametrica 3D: Rete (n) vs Capacità (k)\nFissato Merchant Bias a α=0.5", 
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Nodi della Rete (n)", fontsize=12, labelpad=10)
    ax.set_ylabel("Capacità Iniziale (k)", fontsize=12, labelpad=10)
    ax.set_zlabel("Vita Rete τ", fontsize=12, labelpad=10)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Longevità τ')
    
    # Angolazione custom per vedere bene lo scivolo della k
    ax.view_init(elev=20, azim=-120)
    
    os.makedirs("plots_multi", exist_ok=True)
    out_path = "plots_multi/3D_NK_Alpha05_Surface.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Grafico 3D (n, k, tau) Salvato in: {out_path}")

if __name__ == "__main__":
    generate_3d_nk_plot()
