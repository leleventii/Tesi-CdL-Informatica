import os
import random
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ln_sim.runner_snapshot import load_ln_snapshot
from ln_sim.core import precompute_paths, simulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots_dynamic")

def test_rebalance_strategies():
    print("Caricamento Snapshot LN...")
    G, _ = load_ln_snapshot()
    
    # Prendiamo i Top 200 per test veloce
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:200]
    G_sub = G.subgraph(top_nodes).copy()
    
    # Rinomina i nodi a indici interi 0..N-1
    mapping = {old: i for i, old in enumerate(G_sub.nodes())}
    G_int = nx.relabel_nodes(G_sub, mapping)
    
    n = 200
    k = 50
    alpha = 0.4
    seed = 42
    
    # 10 Merchant
    degrees_int = dict(G_int.degree())
    M = sorted(degrees_int, key=degrees_int.get, reverse=True)[:10]
    
    print("Pre-calcolo dei path (Top 200)...")
    paths = precompute_paths(G_int)
    
    results = {}
    
    print("\n1. Baseline (Nessun Rebalancing)...")
    tau_none, _ = simulate(G_int, k, alpha, M, paths, seed=seed, rebalance_mode="none")
    print(f"  -> tau = {tau_none}")
    results["Nessun Rebalance\n(Baseline)"] = tau_none
    
    print("2. JIT Circular (Probabilistico)...")
    tau_prob, _ = simulate(G_int, k, alpha, M, paths, seed=seed, rebalance_mode="shortest_widest")
    print(f"  -> tau = {tau_prob}")
    results["JIT Circular\n(Shortest-Widest)"] = tau_prob
    
    print("3. Merchant-Driven Rebalance...")
    tau_merch, _ = simulate(G_int, k, alpha, M, paths, seed=seed, rebalance_mode="merchant")
    print(f"  -> tau = {tau_merch}")
    results["Merchant-Driven\nRebalance"] = tau_merch
    
    # Grafico a Barre
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    labels = list(results.keys())
    values = list(results.values())
    colors = ["#e74c3c", "#f1c40f", "#2ecc71"]
    
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_title("Impatto delle Strategie di Rebalancing sulla Longevità della Rete\n(Snapshot Reale: Top 200 Nodi, α = 0.4)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Transazioni Sopravvissute (τ)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "rebalance_strategies_comparison.png")
    fig.savefig(out_path, dpi=300)
    print(f"\nGrafico salvato in: {out_path}")

if __name__ == "__main__":
    test_rebalance_strategies()
