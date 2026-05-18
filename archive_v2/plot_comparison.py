import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_BASE = os.path.join(BASE_DIR, "data_snapshot")
DATA_DIR_REB = os.path.join(BASE_DIR, "data_snapshot_blind_probing")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_comparison")
os.makedirs(PLOTS_DIR, exist_ok=True)

def _load_csv(path):
    if not os.path.exists(path): return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def plot_comparison_SN_A():
    plt.style.use("seaborn-v0_8-whitegrid")
    
    sum_path_base = os.path.join(DATA_DIR_BASE, "SN_A_tau_vs_alpha_summary.csv")
    sum_path_reb = os.path.join(DATA_DIR_REB, "SN_A_tau_vs_alpha_summary.csv")
    
    base = _load_csv(sum_path_base)
    reb = _load_csv(sum_path_reb)
    
    if not base or not reb:
        print("Mancano i dati per SN_A in una delle due cartelle.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Base
    alphas_base = [float(r["param_value"]) for r in base]
    means_base = [float(r["tau_mean"]) for r in base]
    n = base[0]["n_nodes"]
    ax.plot(alphas_base, means_base, color="#34495e", linewidth=2.5, marker="o", label=f"Nessun Rebalancing")
    
    # Reb
    alphas_reb = [float(r["param_value"]) for r in reb]
    means_reb = [float(r["tau_mean"]) for r in reb]
    ax.plot(alphas_reb, means_reb, color="#e74c3c", linewidth=2.5, marker="s", linestyle="--", label=f"Rebalancing Probabilistico (Widest Path su C/2)")
    
    ax.set_yscale("log")
    ax.set_title(f"Impatto del Rebalancing sulla Lightning Network\n(Top {n} nodi, k=50)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Merchant Bias (α)", fontsize=12)
    ax.set_ylabel("τ (Fallimento di Rete) - Scala Log", fontsize=12)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "SN_A_Comparison.png")
    fig.savefig(path, dpi=300)
    print(f"Salvato plot: {path}")

def plot_comparison_SN_B():
    plt.style.use("seaborn-v0_8-whitegrid")
    
    sum_path_base = os.path.join(DATA_DIR_BASE, "SN_B_tau_vs_k_summary.csv")
    sum_path_reb = os.path.join(DATA_DIR_REB, "SN_B_tau_vs_k_summary.csv")
    
    base = _load_csv(sum_path_base)
    reb = _load_csv(sum_path_reb)
    
    if not base or not reb:
        print("Mancano i dati per SN_B in una delle due cartelle.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot solo per alpha = 0.0 e alpha = 0.5 per chiarezza
    alphas_to_plot = ["0.0", "0.5"]
    
    colors_base = {"0.0": "#27ae60", "0.5": "#c0392b"}
    colors_reb = {"0.0": "#2ecc71", "0.5": "#e74c3c"}
    
    def extract_data(summary, target_alpha):
        k_vals = []
        tau_vals = []
        for r in summary:
            if float(r["alpha"]) == float(target_alpha):
                k_vals.append(float(r["param_value"]))
                tau_vals.append(float(r["tau_mean"]))
        return k_vals, tau_vals

    for a in alphas_to_plot:
        k_b, tau_b = extract_data(base, a)
        if k_b:
            ax.plot(k_b, tau_b, color=colors_base[a], marker="o", linewidth=2.5, label=f"No-Reb α={a}")
            
        k_r, tau_r = extract_data(reb, a)
        if k_r:
            ax.plot(k_r, tau_r, color=colors_reb[a], marker="s", linestyle="--", linewidth=2.5, label=f"Prob-Reb α={a}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Legge di Scala (τ vs k) con e senza Rebalancing", fontsize=14, fontweight="bold")
    ax.set_xlabel("Capacità Iniziale (k) - Scala Log", fontsize=12)
    ax.set_ylabel("τ Medio - Scala Log", fontsize=12)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "SN_B_Comparison.png")
    fig.savefig(path, dpi=300)
    print(f"Salvato plot: {path}")

if __name__ == '__main__':
    plot_comparison_SN_A()
    plot_comparison_SN_B()
