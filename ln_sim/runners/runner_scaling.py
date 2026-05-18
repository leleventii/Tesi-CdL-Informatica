"""
runner_scaling.py — τ vs n (Scaling Clique, confronto bias merchant)
======================================================================
Clique K(n), n da 10 a 200 (step 10), k=100, M=⌊√n⌋.
α ∈ {0.0, 0.2, 0.5}: confronto diretto uniforme vs merchant.
20 run per punto, salva CSV in data/, genera grafici in plots/.
Append Mode: skippa automaticamente le configurazioni già calcolate.

Uso:
    python -m ln_sim.runner_scaling
    python -m ln_sim.runner_scaling --force
    python -m ln_sim.runner_scaling --plot-only
"""

import csv
import math
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from ..core import precompute_paths, simulate
from ..graphs import make_clique

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "scaling")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "plots_scaling")

_G = _k = _alpha = _M = _paths = None


def _init_pool(G, k, alpha, M, paths):
    global _G, _k, _alpha, _M, _paths
    _G, _k, _alpha, _M, _paths = G, k, alpha, M, paths


def _worker(args):
    seed, run_id = args
    tau, failed_edge = simulate(_G, _k, _alpha, _M, _paths, seed=seed)
    return run_id, tau


def _run_parallel(G, k, alpha, M, paths, n_runs, desc=""):
    tasks = [(1000 + i, i) for i in range(n_runs)]
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

ALPHAS = [0.0, 0.2, 0.5]

def run_exp_scaling(force=False, n_runs=20, k=100):
    exp_name = "S_scaling"
    sum_path = os.path.join(DATA_DIR, f"{exp_name}_summary.csv")
    raw_path = os.path.join(DATA_DIR, f"{exp_name}_raw.csv")

    if os.path.exists(sum_path) and not force:
        print(f"  ⚠ {sum_path} esiste. Riprendo le simulazioni mancanti (Append Mode).")
    elif force:
        if os.path.exists(sum_path):
            os.remove(sum_path)
        if os.path.exists(raw_path):
            os.remove(raw_path)

    node_counts = list(range(10, 210, 10))  # 10, 20, ..., 200

    print("\n═══ EXP S: τ vs n (Scaling Clique, confronto α) ═══")
    print(f"  n = {node_counts[0]}..{node_counts[-1]} (step 10)")
    print(f"  k={k}, α={ALPHAS}, runs={n_runs}, M=⌊√n⌋")
    print(f"  CPU cores: {cpu_count()}")

    os.makedirs(DATA_DIR, exist_ok=True)

    cols_s = ["n_nodes", "n_edges", "n_merchants", "alpha",
              "tau_mean", "tau_std", "tau_min", "tau_max",
              "tau_median", "ci95_low", "ci95_high", "n_runs",
              "time_sec"]
    cols_r = ["n_nodes", "alpha", "run_id", "tau"]

    # --- Lettura dello stato precedente per SKIPPING ---
    completed_points = set()
    if os.path.exists(sum_path):
        with open(sum_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_points.add((int(r["n_nodes"]), float(r["alpha"])))

    total_points = len(node_counts) * len(ALPHAS)
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
            G = make_clique(n)
            n_edges = G.number_of_edges()
            n_merchants = max(1, int(math.sqrt(n)))
            M = list(range(n_merchants))
            paths = precompute_paths(G)

            for alpha in ALPHAS:
                point += 1

                if (n, alpha) in completed_points:
                    tqdm.write(f"  [SKIPPED] n={n:>3}, α={alpha}")
                    continue

                t0 = time.time()
                desc = f"  [{point}/{total_points}] n={n}, α={alpha}"
                results = _run_parallel(G, k, alpha, M, paths, n_runs, desc)

                taus = [r[1] for r in results]
                arr = np.array(taus, dtype=float)

                # Salva raw
                for run_id, tau in results:
                    rw.writerow({"n_nodes": n, "alpha": alpha,
                                 "run_id": run_id, "tau": tau})
                rf.flush()

                # Stats + summary
                elapsed = time.time() - t0
                mean = np.mean(arr)
                std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
                se = std / np.sqrt(len(arr))

                row = {
                    "n_nodes": n, "n_edges": n_edges,
                    "n_merchants": n_merchants, "alpha": alpha,
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

                tqdm.write(f"  n={n:>3}, α={alpha} → τ={mean:.0f} ± {std:.0f}  "
                           f"[min={np.min(arr):.0f}, max={np.max(arr):.0f}]  "
                           f"[{elapsed:.1f}s]")

    elapsed_total = time.time() - t_global
    print(f"\n  ✓ {sum_path}")
    print(f"  ✓ {raw_path}")
    print(f"  Tempo totale: {elapsed_total/60:.1f} min")


# ═══════════════════════════════════════════════════════════════
#  Grafico — Confronto diretto α sullo stesso plot
# ═══════════════════════════════════════════════════════════════

def plot_scaling():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    DPI = 300
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sum_path = os.path.join(DATA_DIR, "S_scaling_summary.csv")
    if not os.path.exists(sum_path):
        print("  ⚠ Dati scaling non trovati.")
        return

    with open(sum_path) as f:
        rows = list(csv.DictReader(f))

    # Raggruppa per alpha
    by_alpha = {}
    for r in rows:
        a = float(r["alpha"])
        if a not in by_alpha:
            by_alpha[a] = []
        by_alpha[a].append(r)

    colors = {0.0: "#2ecc71", 0.2: "#e67e22", 0.5: "#e74c3c"}
    markers = {0.0: "o", 0.2: "s", 0.5: "D"}
    labels = {
        0.0: "α = 0 (Uniforme)",
        0.2: "α = 0.2 (Bias basso)",
        0.5: "α = 0.5 (Bias medio)",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for alpha in sorted(by_alpha.keys()):
        data = sorted(by_alpha[alpha], key=lambda r: int(r["n_nodes"]))
        ns = np.array([int(r["n_nodes"]) for r in data])
        means = np.array([float(r["tau_mean"]) for r in data])
        ci_lo = np.array([float(r["ci95_low"]) for r in data])
        ci_hi = np.array([float(r["ci95_high"]) for r in data])

        c = colors.get(alpha, "#95a5a6")
        m = markers.get(alpha, "o")
        lb = labels.get(alpha, f"α = {alpha}")

        lw = 3.0 if alpha == 0.0 else 2.0
        zord = 4 if alpha == 0.0 else 2

        ax.fill_between(ns, ci_lo, ci_hi, color=c, alpha=0.12)
        ax.plot(ns, means, color=c, marker=m, markersize=5,
                linewidth=lw, label=lb, zorder=zord)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Numero di nodi n (scala log)", fontsize=13)
    ax.set_ylabel("τ medio (scala log)", fontsize=13)
    ax.set_title("Effetto del Merchant Bias sulla vita della rete\n"
                 "(Clique n nodi, k=100, M=⌊√n⌋ merchant)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "S_tau_vs_n.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp S: Scaling Clique")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    if args.plot_only:
        plot_scaling()
    else:
        run_exp_scaling(force=args.force, n_runs=args.n_runs, k=args.k)
        plot_scaling()
