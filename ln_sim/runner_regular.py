"""
runner_regular.py — Esperimenti su Random Regular Graph
=========================================================
Modulo COMPLETAMENTE SEPARATO dagli esperimenti Clique/Dumbbell.
Dati salvati in data_regular/, grafici in plots_regular/.

Esperimenti:
  - R_A: τ vs α  (stesso schema di Exp A, ma su grafo d-regolare)
  - R_B: τ vs k  (stesso schema di Exp B, ma su grafo d-regolare)

L'obiettivo è dimostrare l'UNIVERSALITÀ del sink-effect:
il fenomeno k² → k e il collasso di τ con α si manifestano
identicamente su topologie diverse dalla Clique.

Uso:
    python3 -m ln_sim.runner_regular              # tutti gli esperimenti
    python3 -m ln_sim.runner_regular --exp R_A     # solo R_A
    python3 -m ln_sim.runner_regular --exp R_B     # solo R_B
    python3 -m ln_sim.runner_regular --plot-only   # solo grafici
    python3 -m ln_sim.runner_regular --force       # sovrascrivi dati
"""

import argparse
import csv
import math
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from .core import precompute_paths, simulate
from .graphs import make_random_regular

# ── Directory SEPARATE ──
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_regular")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots_regular")


# ─────────────────────────────────────────────────
#  Pool globals (stessa tecnica di runner.py)
# ─────────────────────────────────────────────────

_G = _k = _alpha = _M = _paths = None


def _init_pool(G, k, alpha, M, paths):
    global _G, _k, _alpha, _M, _paths
    _G, _k, _alpha, _M, _paths = G, k, alpha, M, paths


def _worker(args):
    seed, run_id = args
    tau, failed_edge = simulate(_G, _k, _alpha, _M, _paths, seed=seed)
    return run_id, tau, failed_edge


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


def _stats(taus):
    arr = np.array(taus, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n)
    return {
        "tau_mean": mean, "tau_std": std,
        "tau_min": np.min(arr), "tau_max": np.max(arr),
        "tau_median": np.median(arr),
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "n_runs": n,
    }


# ═══════════════════════════════════════════════════════════════
#  EXP R_A: τ vs α su Random Regular Graph
# ═══════════════════════════════════════════════════════════════

def run_exp_R_A(force=False):
    """
    Random Regular Graph d-regolare:
    n ∈ {20, 40, 60, 80, 100}, d = 6 (fisso),
    k = 50, M = [√n] merchant, α variabile.
    12 run per configurazione.
    """
    exp_name = "R_A_tau_vs_alpha"
    sum_path = os.path.join(DATA_DIR, f"{exp_name}_summary.csv")
    raw_path = os.path.join(DATA_DIR, f"{exp_name}_raw.csv")

    if os.path.exists(sum_path) and not force:
        print(f"  ⚠ {sum_path} esiste. Riprendo le simulazioni mancanti.")
    elif force:
        if os.path.exists(sum_path): os.remove(sum_path)
        if os.path.exists(raw_path): os.remove(raw_path)

    print("\n═══ EXP R_A: τ vs α (Random Regular, d=6) ═══")

    n_values = [20, 40, 60, 80, 100]
    d = 6  # grado fisso per tutti i nodi
    k = 50
    alphas = [round(a * 0.05, 2) for a in range(13)]  # 0.00..0.60
    n_runs = 12

    os.makedirs(DATA_DIR, exist_ok=True)

    sum_cols = ["param_name", "param_value", "n_nodes", "degree",
                "tau_mean", "tau_std", "tau_min", "tau_max", "tau_median",
                "ci95_low", "ci95_high", "n_runs"]
    raw_cols = ["param_value", "n_nodes", "degree", "run_id", "tau"]

    # Leggi punti completati
    completed = set()
    if os.path.exists(sum_path):
        with open(sum_path, "r") as f:
            for r in csv.DictReader(f):
                completed.add((int(r["n_nodes"]), float(r["param_value"])))

    write_headers = not os.path.exists(sum_path)

    with open(sum_path, "a", newline="") as sf, \
         open(raw_path, "a", newline="") as rf:
        sw = csv.DictWriter(sf, fieldnames=sum_cols)
        if write_headers: sw.writeheader()
        rw = csv.DictWriter(rf, fieldnames=raw_cols)
        if write_headers: rw.writeheader()

        for n in n_values:
            print(f"\n  ── n = {n}, d = {d} ──")
            G = make_random_regular(n, d)
            M = list(range(max(1, int(math.sqrt(n)))))
            paths = precompute_paths(G)

            for alpha in tqdm(alphas, desc=f"  α (n={n})", colour="cyan"):
                if (n, alpha) in completed:
                    tqdm.write(f"  [SKIPPED] n={n}, α={alpha}")
                    continue

                results = _run_parallel(G, k, alpha, M, paths, n_runs)
                taus = [r[1] for r in results]

                for run_id, tau, _ in results:
                    rw.writerow({"param_value": alpha, "n_nodes": n,
                                 "degree": d, "run_id": run_id, "tau": tau})
                rf.flush()

                stats = _stats(taus)
                row = {"param_name": "alpha", "param_value": alpha,
                       "n_nodes": n, "degree": d}
                row.update({k_: f"{v:.2f}" if isinstance(v, float) else v
                            for k_, v in stats.items()})
                sw.writerow(row)
                sf.flush()

                tqdm.write(f"    α={alpha:.2f} (n={n}) → τ={stats['tau_mean']:.0f} "
                           f"± {stats['tau_std']:.0f}")

    print(f"  ✓ Salvato: {sum_path}")
    print(f"  ✓ Salvato: {raw_path}")


# ═══════════════════════════════════════════════════════════════
#  EXP R_B: τ vs k su Random Regular Graph
# ═══════════════════════════════════════════════════════════════

def run_exp_R_B(force=False):
    """
    Random Regular Graph d-regolare:
    n ∈ {20, 40, 60, 80, 100}, d = 6,
    α ∈ {0.0, 0.25, 0.5}, k variabile.
    12 run per configurazione.
    """
    exp_name = "R_B_tau_vs_k"
    sum_path = os.path.join(DATA_DIR, f"{exp_name}_summary.csv")
    raw_path = os.path.join(DATA_DIR, f"{exp_name}_raw.csv")
    slope_path = os.path.join(DATA_DIR, f"{exp_name}_slopes.csv")

    if os.path.exists(sum_path) and not force:
        print(f"  ⚠ {sum_path} esiste già — usa --force. Salto.")
        return

    print("\n═══ EXP R_B: τ vs k (Random Regular, d=6) ═══")

    n_values = [20, 40, 60, 80, 100]
    d = 6
    alphas = [0.0, 0.25, 0.5]
    k_values = [10, 20, 50, 100, 200]
    n_runs = 12

    os.makedirs(DATA_DIR, exist_ok=True)

    sum_cols = ["param_name", "param_value", "n_nodes", "degree",
                "tau_mean", "tau_std", "tau_min", "tau_max", "tau_median",
                "ci95_low", "ci95_high", "n_runs", "alpha"]
    raw_cols = ["param_value", "n_nodes", "degree", "run_id", "tau", "alpha"]

    slope_data = []

    with open(sum_path, "w", newline="") as sf, \
         open(raw_path, "w", newline="") as rf:
        sw = csv.DictWriter(sf, fieldnames=sum_cols)
        sw.writeheader()
        rw = csv.DictWriter(rf, fieldnames=raw_cols)
        rw.writeheader()

        for alpha in alphas:
            print(f"\n  ── α = {alpha} ──")
            for n in n_values:
                G = make_random_regular(n, d)
                M = list(range(max(1, int(math.sqrt(n)))))
                paths = precompute_paths(G)
                means_for_slope = []

                for k in tqdm(k_values, desc=f"  k (α={alpha}, n={n})", colour="cyan"):
                    results = _run_parallel(G, k, alpha, M, paths, n_runs)
                    taus = [r[1] for r in results]

                    for run_id, tau, _ in results:
                        rw.writerow({"param_value": k, "n_nodes": n, "degree": d,
                                     "run_id": run_id, "tau": tau, "alpha": alpha})
                    rf.flush()

                    stats = _stats(taus)
                    means_for_slope.append(stats["tau_mean"])
                    row = {"param_name": "k", "param_value": k,
                           "n_nodes": n, "degree": d, "alpha": alpha}
                    row.update({k_: f"{v:.2f}" if isinstance(v, float) else v
                                for k_, v in stats.items()})
                    sw.writerow(row)
                    sf.flush()

                log_k = np.log(k_values)
                log_tau = np.log(np.maximum(means_for_slope, 1e-9))
                slope = np.polyfit(log_k, log_tau, 1)[0]
                slope_data.append({"alpha": alpha, "n": n, "degree": d, "slope": slope})
                print(f"  Slope log-log (α={alpha}, n={n}): {slope:.3f}")

    with open(slope_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "n", "degree", "slope"])
        w.writeheader()
        for dd in slope_data:
            dd_fmt = dd.copy()
            dd_fmt["slope"] = f"{dd['slope']:.4f}"
            w.writerow(dd_fmt)

    print(f"  ✓ Salvato: {sum_path}")
    print(f"  ✓ Salvato: {raw_path}")
    print(f"  ✓ Salvato: {slope_path}")


# ═══════════════════════════════════════════════════════════════
#  Grafici
# ═══════════════════════════════════════════════════════════════

def _load_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _save(fig, name):
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_R_A():
    """τ vs α con CI 95% per vari n su Random Regular Graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    sum_path = os.path.join(DATA_DIR, "R_A_tau_vs_alpha_summary.csv")
    if not os.path.exists(sum_path):
        print("  ⚠ Dati R_A non trovati, salto plot.")
        return

    summary = _load_csv(sum_path)
    by_n = {}
    for r in summary:
        n = int(r["n_nodes"])
        if n not in by_n:
            by_n[n] = {"alphas": [], "means": [], "ci_low": [], "ci_high": []}
        by_n[n]["alphas"].append(float(r["param_value"]))
        by_n[n]["means"].append(float(r["tau_mean"]))
        by_n[n]["ci_low"].append(float(r["ci95_low"]))
        by_n[n]["ci_high"].append(float(r["ci95_high"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    n_sorted = sorted(by_n.keys())

    for i, n in enumerate(n_sorted):
        color = cmap(i / max(1, len(n_sorted) - 1))
        d = by_n[n]
        ax.fill_between(d["alphas"], d["ci_low"], d["ci_high"],
                        color=color, alpha=0.1)
        ax.plot(d["alphas"], d["means"], color=color, linewidth=2,
                marker="o", markersize=4, label=f"n={n}")

    ax.set_yscale("log")
    ax.set_title("Tempo di fallimento τ al variare del bias α\n"
                 "(Random Regular Graph, d=6, k=50, M=[√n] merchant)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Probabilità bias merchant α", fontsize=12)
    ax.set_ylabel("τ medio (scala logaritmica)", fontsize=12)
    ax.legend(fontsize=10, title="Dimensione Rete")
    fig.tight_layout()
    _save(fig, "R_A_tau_vs_alpha.png")


def plot_R_B():
    """τ vs k in scala log-log su Random Regular Graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    sum_path = os.path.join(DATA_DIR, "R_B_tau_vs_k_summary.csv")
    slope_path = os.path.join(DATA_DIR, "R_B_tau_vs_k_slopes.csv")

    if not os.path.exists(sum_path):
        print("  ⚠ Dati R_B non trovati, salto plot.")
        return

    summary = _load_csv(sum_path)
    data = {}
    for r in summary:
        a = float(r["alpha"])
        n = int(r["n_nodes"])
        key = (a, n)
        if key not in data:
            data[key] = {"k": [], "mean": []}
        data[key]["k"].append(float(r["param_value"]))
        data[key]["mean"].append(float(r["tau_mean"]))

    slopes = {}
    if os.path.exists(slope_path):
        for r in _load_csv(slope_path):
            slopes[(float(r["alpha"]), int(r["n"]))] = float(r["slope"])

    colors = {0.0: plt.get_cmap("Blues"), 0.25: plt.get_cmap("Greens"),
              0.5: plt.get_cmap("Reds")}

    fig, ax = plt.subplots(figsize=(10, 6))
    all_n = sorted(set(k[1] for k in data.keys()))

    for (alpha, n), d in sorted(data.items()):
        k_arr = np.array(d["k"])
        m_arr = np.array(d["mean"])
        idx = all_n.index(n)
        intens = 0.4 + 0.6 * (idx + 1) / max(1, len(all_n))
        col = colors.get(alpha, plt.get_cmap("Greys"))(intens)

        ax.plot(k_arr, m_arr, color=col, marker="o", markersize=4,
                linewidth=2, label=f"α={alpha}, n={n}")

        if (alpha, n) in slopes:
            s = slopes[(alpha, n)]
            log_k = np.log(k_arr)
            log_m = np.log(np.maximum(m_arr, 1e-9))
            intercept = np.mean(log_m - s * log_k)
            fit_tau = np.exp(intercept + s * np.log(k_arr))
            ax.plot(k_arr, fit_tau, color=col, linestyle="--",
                    linewidth=1, alpha=0.5)

    # Annotazione slopes
    annot_parts = []
    for a in sorted({k[0] for k in slopes.keys()}):
        line = f"α={a}: slopes ≈ ["
        sl = [f"{slopes[(a, n)]:.2f}" for n in all_n if (a, n) in slopes]
        line += ", ".join(sl) + "]"
        annot_parts.append(line)

    if annot_parts:
        ax.text(0.05, 0.95, "\n".join(annot_parts),
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Scaling di τ con k (Random Regular Graph, d=6)\n"
                 "Transizione O(k²) → O(k) con merchant bias",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Capacità k (scala log)", fontsize=12)
    ax.set_ylabel("τ medio (scala log)", fontsize=12)
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    _save(fig, "R_B_tau_vs_k_loglog.png")


def plot_all_regular():
    print("\n═══ Generazione grafici Random Regular ═══")
    plot_R_A()
    plot_R_B()
    print("  Done!")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esperimenti Random Regular Graph")
    parser.add_argument("--exp", choices=["R_A", "R_B", "all"], default="all",
                        help="Esperimento da eseguire")
    parser.add_argument("--plot-only", action="store_true",
                        help="Genera solo i grafici dai CSV esistenti")
    parser.add_argument("--force", action="store_true",
                        help="Sovrascrivi CSV esistenti")

    args = parser.parse_args()

    if args.plot_only:
        plot_all_regular()
    else:
        dispatch = {"R_A": run_exp_R_A, "R_B": run_exp_R_B}
        exps = ["R_A", "R_B"] if args.exp == "all" else [args.exp]

        t0 = time.time()
        for e in exps:
            dispatch[e](force=args.force)
        elapsed = time.time() - t0
        print(f"\n  Completato in {elapsed:.0f}s")

        plot_all_regular()
