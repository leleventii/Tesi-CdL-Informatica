import os
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ln_sim.graphs import make_clique
from ln_sim.core import precompute_paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots_dynamic")

def simulate_track_wealth(n=100, k=50, alpha=0.4, seed=42):
    print(f"Esecuzione simulazione Clique N={n}, alpha={alpha}...")
    G = make_clique(n)
    M = list(range(int(np.sqrt(n)))) # Merchant
    M_set = set(M)
    C_set = set(G.nodes()).difference(M_set)
    
    paths = precompute_paths(G)
    random.seed(seed)
    
    nodes = list(G.nodes())
    bal = np.zeros((n, n), dtype=np.int64)
    
    for u, v in G.edges():
        bal[u, v] = k
        bal[v, u] = k
        
    merchant_wealth = []
    client_wealth = []
    
    sample_rate = 100
    tau = 0
    
    mw = sum(bal[m, v] for m in M_set for v in G.neighbors(m))
    cw = sum(bal[c, v] for c in C_set for v in G.neighbors(c))
    merchant_wealth.append(mw)
    client_wealth.append(cw)

    while True:
        s = random.choice(nodes)
        if random.random() < alpha:
            d = random.choice(M)
        else:
            d = s
            while d == s:
                d = random.choice(nodes)
                
        if s == d:
            tau += 1
            continue
            
        path = random.choice(paths[(s, d)])
        
        can_route = True
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if bal[u, v] < 1:
                can_route = False
                break
                
        if can_route:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                bal[u, v] -= 1
                bal[v, u] += 1
            tau += 1
            
            if tau % sample_rate == 0:
                mw = sum(bal[m, v] for m in M_set for v in G.neighbors(m))
                cw = sum(bal[c, v] for c in C_set for v in G.neighbors(c))
                merchant_wealth.append(mw)
                client_wealth.append(cw)
        else:
            break
            
    print(f"-> Fallimento a tau={tau}")
    return merchant_wealth, client_wealth, tau, sample_rate

def plot_wealth_transfer():
    alpha = 0.4
    mw, cw, tau, sr = simulate_track_wealth(n=100, k=50, alpha=alpha, seed=42)
        
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_axis = np.arange(len(mw)) * sr
    
    ax.plot(t_axis, mw, color="#e74c3c", linewidth=3, label="Ricchezza Merchant (Hub)")
    ax.plot(t_axis, cw, color="#3498db", linewidth=3, label="Ricchezza Client (User)")
    
    ax.set_title(f"Trasferimento di Ricchezza nel Tempo (Bias Asimmetrico: α = {alpha})\nDimostrazione del Sink Effect", fontsize=14, fontweight="bold")
    ax.set_xlabel("Transazioni elaborate (t)", fontsize=12)
    ax.set_ylabel("Somma totale dei bilanci in uscita (Satoshi)", fontsize=12)
    
    # Text box per spiegazione
    text_str = (
        f"Collasso Rete a t={tau}\n\n"
        "La rete muore molto prima che i Merchant\n"
        "possano assorbire tutta la ricchezza,\n"
        "poiché i ponti diretti si esauriscono."
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.65, 0.5, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props)
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)

    fig.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "wealth_transfer_single.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nGrafico salvato in: {out_path}")

if __name__ == "__main__":
    plot_wealth_transfer()
