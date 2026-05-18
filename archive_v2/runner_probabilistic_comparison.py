import os
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ln_sim.runner_snapshot import load_ln_snapshot
from ln_sim.core import simulate

class DynamicPathsCache:
    def __init__(self, G, max_cache_size=200000):
        self.G = G
        self.max_cache_size = max_cache_size
        self._cache = {}

    def __getitem__(self, key):
        s, d = key
        if key not in self._cache:
            try:
                paths = list(nx.all_shortest_paths(self.G, source=s, target=d))
                self._cache[key] = paths
                if len(self._cache) > self.max_cache_size:
                    keys_to_remove = list(self._cache.keys())[:int(self.max_cache_size * 0.2)]
                    for k in keys_to_remove:
                        del self._cache[k]
            except nx.NetworkXNoPath:
                self._cache[key] = []
        return self._cache[key]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots_comparison")

def run_probabilistic_comparison(n_nodes=500):
    print(f"Caricamento Snapshot LN e estrazione Top {n_nodes} nodi...")
    G, _ = load_ln_snapshot()
    
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:n_nodes]
    G_sub = G.subgraph(top_nodes).copy()
    
    mapping = {old: i for i, old in enumerate(G_sub.nodes())}
    G_int = nx.relabel_nodes(G_sub, mapping)
    
    # M = sqrt(N)
    n_merchants = int(np.sqrt(n_nodes))
    degrees_int = dict(G_int.degree())
    M = sorted(degrees_int, key=degrees_int.get, reverse=True)[:n_merchants]
    
    print("Inizializzazione cache percussiva dei path...")
    paths = DynamicPathsCache(G_int, max_cache_size=200000)
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    k = 50
    n_runs = 10
    
    tau_base_means = []
    tau_prob_means = []
    
    print("\nAvvio simulazioni (10 run per bias)...")
    for a in alphas:
        print(f"  Simulazione alpha = {a}...")
        tb_list = []
        tp_list = []
        
        for run_id in range(n_runs):
            seed = 1000 + run_id
            
            # Baseline
            tb, _ = simulate(G_int, k, a, M, paths, seed=seed, rebalance_mode="none")
            tb_list.append(tb)
            
            # Probabilistic Rebalance (C/2)
            tp, _ = simulate(G_int, k, a, M, paths, seed=seed, rebalance_mode="shortest_widest")
            tp_list.append(tp)
            
        tb_mean = np.mean(tb_list)
        tp_mean = np.mean(tp_list)
        
        tau_base_means.append(tb_mean)
        tau_prob_means.append(tp_mean)
        
        print(f"    Media Baseline: {tb_mean:.1f} | Media Probabilistico (C/2): {tp_mean:.1f}")
        
    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(alphas, tau_base_means, color="#34495e", linewidth=2.5, marker="o", label="Nessun Rebalancing")
    ax.plot(alphas, tau_prob_means, color="#e74c3c", linewidth=2.5, marker="^", linestyle="-.", label="Probabilistico (C/2)")
    
    ax.set_yscale("log")
    ax.set_title(f"Impatto del Rebalancing Probabilistico sulla Lightning Network\n(Snapshot Reale: Top {n_nodes} nodi, k=50)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Merchant Bias (α)", fontsize=12)
    ax.set_ylabel("τ (Fallimento di Rete) - Scala Log", fontsize=12)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "probabilistic_comparison_top500.png")
    fig.savefig(out_path, dpi=300)
    print(f"\nGrafico salvato in: {out_path}")

if __name__ == "__main__":
    run_probabilistic_comparison(500)
