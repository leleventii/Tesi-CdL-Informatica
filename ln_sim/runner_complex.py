"""
runner_complex.py — τ vs n (Topologie Complesse)
======================================================================
Confronta Barabási-Albert, Erdős-Rényi e Random Regular tenendo
costante il grado medio (d=10), k=100, M=⌊√n⌋, a un alpha fisso (es. 0.5).
Salva CSV in data/ e plot in plots/.

Uso:
    python -m ln_sim.runner_complex
    python -m ln_sim.runner_complex --force
    python -m ln_sim.runner_complex --plot-only
"""

import csv
import math
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from .core import precompute_paths, simulate
from .graphs import make_barabasi_albert, make_erdos_renyi, make_random_regular

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")

_G = _k = _alpha = _M = _paths = None


def _init_pool(G, k, alpha, M, paths):
    global _G, _k, _alpha, _M, _paths
    _G, _k, _alpha, _M, _paths = G, k, alpha, M, paths


def _worker(args):
    seed, run_id = args
    tau, failed_edge = simulate(_G, _k, _alpha, _M, _paths, seed=seed)
    return run_id, tau


def _run_parallel(G, k, alpha, M, paths, n_runs, desc=""):
    tasks = [(2000 + i, i) for i in range(n_runs)]
    with Pool(
        processes=cpu_count(),
        initializer=_init_pool,
        initargs=(G, k, alpha, M, paths),
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(_worker, tasks),
            total=n_runs, desc=desc, leave=False, colour="green",
        ))
    results.sort(key=lambda x: x[0])
    return results


# ═══════════════════════════════════════════════════════════════
#  Simulazione
# ═══════════════════════════════════════════════════════════════

TOPOLOGIES = ["barabasi_albert", "erdos_renyi", "random_regular"]

def get_graph(topo_name, n, d):
    if topo_name == "barabasi_albert":
        return make_barabasi_albert(n, d)
    elif topo_name == "erdos_renyi":
        return make_erdos_renyi(n, d)
    elif topo_name == "random_regular":
        return make_random_regular(n, d)
    else:
        raise ValueError(f"Topologia non supportata: {topo_name}")

def run_exp_complex(force=False, n_runs=20, k=100, alpha=0.5, d=10):
    exp_name = "C_complex"
    sum_path = os.path.join(DATA_DIR, f"{exp_name}_summary.csv")
    raw_path = os.path.join(DATA_DIR, f"{exp_name}_raw.csv")

    if os.path.exists(sum_path) and not force:
        print(f"  ⚠ {sum_path} esiste. Riprendo (Append Mode).")
    elif force:
        if os.path.exists(sum_path): os.remove(sum_path)
        if os.path.exists(raw_path): os.remove(raw_path)

    node_counts = [50, 100, 150, 200, 300, 400, 500]

    print("\n═══ EXP C: τ vs n (Topologie Complesse) ═══")
    print(f"  n = {node_counts}")
    print(f"  Tipi = {TOPOLOGIES}")
    print(f"  k={k}, α={alpha}, runs={n_runs}, d={d}, M=⌊√n⌋")
    print(f"  CPU cores: {cpu_count()}")

    os.makedirs(DATA_DIR, exist_ok=True)

    cols_s = ["topology", "n_nodes", "n_edges", "n_merchants",
              "tau_mean", "tau_std", "tau_min", "tau_max",
              "tau_median", "ci95_low", "ci95_high", "n_runs",
              "time_sec"]
    cols_r = ["topology", "n_nodes", "run_id", "tau"]

    completed_points = set()
    if os.path.exists(sum_path):
        with open(sum_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_points.add((r["topology"], int(r["n_nodes"])))

    total_points = len(node_counts) * len(TOPOLOGIES)
    t_global = time.time()
    write_headers = not os.path.exists(sum_path)

    with open(sum_path, "a", newline="") as sf, \
         open(raw_path, "a", newline="") as rf:
        sw = csv.DictWriter(sf, fieldnames=cols_s)
        if write_headers: sw.writeheader()

        rw = csv.DictWriter(rf, fieldnames=cols_r)
        if write_headers: rw.writeheader()

        point = 0
        for n in node_counts:
            n_merchants = max(1, int(math.sqrt(n)))
            M = list(range(n_merchants))
            
            for topo in TOPOLOGIES:
                point += 1
                if (topo, n) in completed_points:
                    tqdm.write(f"  [SKIPPED] topo={topo}, n={n:>3}")
                    continue

                t0 = time.time()
                desc = f"  [{point}/{total_points}] {topo[:3].upper()} n={n}"
                
                # Generazione Grafo
                G = get_graph(topo, n, d)
                # Siccome i nodi possono essere rinumerati, ci assicuriamo che M e paths siano coerenti
                nodes = list(G.nodes())
                M_actual = nodes[:n_merchants] # Prendiamo i primi nodi (che nel BA sono i veri HUB)
                paths = precompute_paths(G)
                
                n_edges = G.number_of_edges()

                results = _run_parallel(G, k, alpha, M_actual, paths, n_runs, desc)

                taus = [r[1] for r in results]
                arr = np.array(taus, dtype=float)

                for run_id, tau in results:
                    rw.writerow({"topology": topo, "n_nodes": n,
                                 "run_id": run_id, "tau": tau})
                rf.flush()

                elapsed = time.time() - t0
                mean = np.mean(arr)
                std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
                se = std / np.sqrt(len(arr))

                row = {
                    "topology": topo, "n_nodes": n, "n_edges": n_edges,
                    "n_merchants": n_merchants,
                    "tau_mean": f"{mean:.2f}", "tau_std": f"{std:.2f}",
                    "tau_min": f"{np.min(arr):.2f}",
                    "tau_max": f"{np.max(arr):.2f}",
                    "tau_median": f"{np.median(arr):.2f}",
                    "ci95_low": f"{mean - 1.96 * se:.2f}",
                    "ci95_high": f"{mean + 1.96 * se:.2f}",
                    "n_runs": len(arr),
                    "time_sec": f"{elapsed:.1f}",
                }
                sw.writerow(row)
                sf.flush()

                tqdm.write(f"  {topo[:3].upper()} n={n:>3} → τ={mean:.0f} ± {std:.0f}  "
                           f"[{elapsed:.1f}s]")

    elapsed_total = time.time() - t_global
    print(f"\n  ✓ {sum_path}")
    print(f"  ✓ {raw_path}")
    print(f"  Tempo totale: {elapsed_total/60:.1f} min")


# ═══════════════════════════════════════════════════════════════
#  Grafico
# ═══════════════════════════════════════════════════════════════

def plot_complex():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    DPI = 300
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sum_path = os.path.join(DATA_DIR, "C_complex_summary.csv")
    if not os.path.exists(sum_path):
        print("  ⚠ Dati complex non trovati.")
        return

    with open(sum_path) as f:
        rows = list(csv.DictReader(f))

    by_topo = {}
    for r in rows:
        t = r["topology"]
        if t not in by_topo:
            by_topo[t] = []
        by_topo[t].append(r)

    colors = {
        "barabasi_albert": "#e74c3c",  # Rosso LN
        "erdos_renyi": "#3498db",      # Blu casulità pura
        "random_regular": "#2ecc71"    # Verde controllo
    }
    markers = {
        "barabasi_albert": "D", 
        "erdos_renyi": "s", 
        "random_regular": "o"
    }
    labels = {
        "barabasi_albert": "Scale-Free (Barabási-Albert) - Hub centrali",
        "erdos_renyi": "Random Graph (Erdős-Rényi)",
        "random_regular": "Random Regular (Grado fisso)"
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for topo in ["barabasi_albert", "erdos_renyi", "random_regular"]:
        if topo not in by_topo:
            continue
            
        data = sorted(by_topo[topo], key=lambda r: int(r["n_nodes"]))
        ns = np.array([int(r["n_nodes"]) for r in data])
        means = np.array([float(r["tau_mean"]) for r in data])
        ci_lo = np.array([float(r["ci95_low"]) for r in data])
        ci_hi = np.array([float(r["ci95_high"]) for r in data])

        c = colors.get(topo, "#95a5a6")
        m = markers.get(topo, "o")
        lb = labels.get(topo, topo)

        ax.fill_between(ns, ci_lo, ci_hi, color=c, alpha=0.15)
        ax.plot(ns, means, color=c, marker=m, markersize=6,
                linewidth=2.5, label=lb)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Numero di nodi n (scala log)", fontsize=13)
    ax.set_ylabel("τ medio (scala log)", fontsize=13)
    ax.set_title("Robustezza topologica al Merchant Bias (α=0.5)\n"
                 "(n nodi, grado medio d=10, k=100, M=⌊√n⌋ merchant)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "C_tau_vs_n_complex.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp C: Complex Networks")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--d", type=int, default=10)
    args = parser.parse_args()

    if args.plot_only:
        plot_complex()
    else:
        run_exp_complex(force=args.force, n_runs=args.n_runs, k=args.k, alpha=args.alpha, d=args.d)
        plot_complex()
