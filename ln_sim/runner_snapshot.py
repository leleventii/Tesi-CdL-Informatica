"""
runner_snapshot.py — Esperimenti su sottografo della vera Lightning Network
=============================================================================
Modulo COMPLETAMENTE SEPARATO.
Dati in data_snapshot/, grafici in plots_snapshot/.

Carica lo snapshot JSON della rete LN reale, estrae un sottografo
dei top nodi per grado (gli hub/exchange), e ci fa girare Algorithm 1
per verificare che il sink-effect (k² → k) si replica su topologia reale.

Uso:
    python3 -m ln_sim.runner_snapshot                    # analisi + esperimenti + grafici
    python3 -m ln_sim.runner_snapshot --analyze-only     # solo analisi del grafo
    python3 -m ln_sim.runner_snapshot --exp SN_A         # solo τ vs α
    python3 -m ln_sim.runner_snapshot --exp SN_B         # solo τ vs k
    python3 -m ln_sim.runner_snapshot --plot-only        # solo grafici
    python3 -m ln_sim.runner_snapshot --force            # sovrascrivi
    python3 -m ln_sim.runner_snapshot --top-n 80         # dimensione sottografo (default 60)
"""

import argparse
import csv
import json
import math
import os
import time
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np
from tqdm import tqdm

from .core import precompute_paths, simulate

# ── Directory SEPARATE ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data_snapshot")
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots_snapshot")
SNAPSHOT_PATH = os.path.join(DATA_DIR, "snapshot_260323.json") # Il file JSON va sempre tenuto in data_snapshot base!

def set_output_dirs(suffix):
    global DATA_DIR, PLOTS_DIR
    if suffix:
        DATA_DIR = os.path.join(BASE_DIR, "..", f"data_snapshot_{suffix}")
        PLOTS_DIR = os.path.join(BASE_DIR, "..", f"plots_snapshot_{suffix}")
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        print(f"  [Output isolato in: {DATA_DIR} e {PLOTS_DIR}]")


# ─────────────────────────────────────────────────
#  Caricamento e parsing dello snapshot LN
# ─────────────────────────────────────────────────

def load_ln_snapshot(path=None):
    """
    Carica lo snapshot JSON della Lightning Network e costruisce
    un grafo NetworkX pesato con le informazioni rilevanti.

    Filtra:
    - Solo canali con ENTRAMBE le policy attive (bidirezionali)
    - Solo canali NON disabilitati su entrambi i lati

    Returns
    -------
    G : nx.Graph
        Grafo non orientato. Attributi arco: 'capacity' (satoshi).
    alias_map : dict
        pub_key → alias per i nodi.
    """
    if path is None:
        path = SNAPSHOT_PATH

    print(f"  Caricamento snapshot: {path}")
    with open(path, "r") as f:
        data = json.load(f)

    # Mappa alias
    alias_map = {}
    for n in data["nodes"]:
        alias_map[n["pub_key"]] = n.get("alias", "???")

    # Costruisci grafo da canali attivi bidirezionali
    G = nx.Graph()
    skipped_disabled = 0
    skipped_no_policy = 0

    for e in data["edges"]:
        p1 = e["node1_policy"]
        p2 = e["node2_policy"]

        if p1 is None or p2 is None:
            skipped_no_policy += 1
            continue
        if p1.get("disabled", True) or p2.get("disabled", True):
            skipped_disabled += 1
            continue

        n1 = e["node1_pub"]
        n2 = e["node2_pub"]
        cap = int(e["capacity"])

        # Se esiste già un canale tra questi nodi, somma la capacity
        if G.has_edge(n1, n2):
            G[n1][n2]["capacity"] += cap
            G[n1][n2]["n_channels"] += 1
        else:
            G.add_edge(n1, n2, capacity=cap, n_channels=1)

    print(f"  Nodi nel grafo: {G.number_of_nodes()}")
    print(f"  Archi nel grafo: {G.number_of_edges()}")
    print(f"  Canali scartati (no policy): {skipped_no_policy}")
    print(f"  Canali scartati (disabled): {skipped_disabled}")

    return G, alias_map


def extract_subgraph(G, alias_map, top_n=60):
    """
    Estrae un sottografo connesso dei top_n nodi per grado.

    Strategia:
    1. Prendi i top_n nodi per grado
    2. Induci il sottografo
    3. Prendi la componente connessa più grande
    4. Ri-etichetta i nodi con indici interi 0..n-1

    Returns
    -------
    G_sub : nx.Graph
        Sottografo con nodi 0..n-1.
    node_info : list of dict
        Info per ogni nodo: indice, pub_key, alias, grado originale.
    """
    # Top nodi per grado
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]

    # Induci sottografo
    G_sub = G.subgraph(top_nodes).copy()

    # Componente connessa più grande
    components = sorted(nx.connected_components(G_sub), key=len, reverse=True)
    G_cc = G_sub.subgraph(components[0]).copy()

    print(f"\n  Sottografo estratto:")
    print(f"    Top {top_n} nodi richiesti")
    print(f"    Componente connessa più grande: {G_cc.number_of_nodes()} nodi, "
          f"{G_cc.number_of_edges()} archi")

    # Ri-etichetta con interi
    old_nodes = sorted(G_cc.nodes(), key=lambda x: degrees[x], reverse=True)
    mapping = {old: i for i, old in enumerate(old_nodes)}
    G_int = nx.relabel_nodes(G_cc, mapping)

    # Info nodi
    node_info = []
    for old, new_id in sorted(mapping.items(), key=lambda x: x[1]):
        node_info.append({
            "id": new_id,
            "pub_key": old[:16] + "...",
            "alias": alias_map.get(old, "???")[:30],
            "degree_original": degrees[old],
            "degree_subgraph": G_int.degree(new_id),
        })

    # Stampa i primi 10
    print(f"\n  Top 10 nodi nel sottografo:")
    print(f"    {'ID':>4}  {'Grado_orig':>10}  {'Grado_sub':>9}  Alias")
    for ni in node_info[:10]:
        print(f"    {ni['id']:>4}  {ni['degree_original']:>10}  "
              f"{ni['degree_subgraph']:>9}  {ni['alias']}")

    return G_int, node_info


def analyze_snapshot(top_n=60):
    """Analisi completa dello snapshot senza eseguire simulazioni."""
    G, alias_map = load_ln_snapshot()
    G_sub, node_info = extract_subgraph(G, alias_map, top_n=top_n)

    print(f"\n  Proprietà del sottografo:")
    print(f"    Nodi: {G_sub.number_of_nodes()}")
    print(f"    Archi: {G_sub.number_of_edges()}")
    print(f"    Densità: {nx.density(G_sub):.4f}")
    print(f"    Diametro: {nx.diameter(G_sub) if nx.is_connected(G_sub) else 'non connesso'}")
    print(f"    Grado medio: {2 * G_sub.number_of_edges() / G_sub.number_of_nodes():.1f}")

    # Distribuzione gradi nel sottografo
    degs = sorted([d for _, d in G_sub.degree()], reverse=True)
    print(f"    Gradi: min={min(degs)}, max={max(degs)}, mediano={degs[len(degs)//2]}")

    # Salva info nodi
    os.makedirs(DATA_DIR, exist_ok=True)
    info_path = os.path.join(DATA_DIR, "subgraph_nodes.csv")
    with open(info_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "pub_key", "alias",
                                           "degree_original", "degree_subgraph"])
        w.writeheader()
        for ni in node_info:
            w.writerow(ni)
    print(f"\n  ✓ {info_path}")

    return G_sub, node_info


# ─────────────────────────────────────────────────
#  Pool globals
# ─────────────────────────────────────────────────

_G = _k = _alpha = _M = _paths = _rebalance = None


def _init_pool(G, k, alpha, M, paths, rebalance):
    global _G, _k, _alpha, _M, _paths, _rebalance
    _G, _k, _alpha, _M, _paths, _rebalance = G, k, alpha, M, paths, rebalance


def _worker(args):
    seed, run_id = args
    tau, failed_edge = simulate(_G, _k, _alpha, _M, _paths, seed=seed, rebalance_active=_rebalance)
    return run_id, tau, failed_edge


def _run_parallel(G, k, alpha, M, paths, n_runs, rebalance=False, desc=""):
    tasks = [(1000 + i, i) for i in range(n_runs)]
    with Pool(
        processes=cpu_count(),
        initializer=_init_pool,
        initargs=(G, k, alpha, M, paths, rebalance),
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
#  EXP SN_A: τ vs α sulla LN reale
# ═══════════════════════════════════════════════════════════════

def run_exp_SN_A(force=False, top_n=60, rebalance=False):
    """
    Sottografo top-N della vera LN.
    Merchant = i top √n nodi per grado (gli hub/exchange).
    k = 50, α variabile.
    """
    exp_name = "SN_A_tau_vs_alpha"
    sum_path = os.path.join(DATA_DIR, f"{exp_name}_summary.csv")
    raw_path = os.path.join(DATA_DIR, f"{exp_name}_raw.csv")

    if os.path.exists(sum_path) and not force:
        print(f"  ⚠ {sum_path} esiste — usa --force. Salto.")
        return

    print(f"\n═══ EXP SN_A: τ vs α (LN Reale, top {top_n} nodi) ═══")

    G, alias_map = load_ln_snapshot()
    G_sub, node_info = extract_subgraph(G, alias_map, top_n=top_n)

    n = G_sub.number_of_nodes()
    k = 50
    # I merchant sono i nodi con grado più alto nel sottografo (già ordinati per grado)
    n_merchants = max(1, int(math.sqrt(n)))
    M = list(range(n_merchants))
    alphas = [round(a * 0.05, 2) for a in range(13)]  # 0.00..0.60
    n_runs = 12

    print(f"  n = {n}, k = {k}, M = {n_merchants} merchant (top hub)")
    print(f"  Merchant: {[node_info[m]['alias'] for m in M]}")

    paths = precompute_paths(G_sub)

    os.makedirs(DATA_DIR, exist_ok=True)

    sum_cols = ["param_name", "param_value", "n_nodes", "n_edges",
                "tau_mean", "tau_std", "tau_min", "tau_max", "tau_median",
                "ci95_low", "ci95_high", "n_runs"]
    raw_cols = ["param_value", "n_nodes", "run_id", "tau"]

    with open(sum_path, "w", newline="") as sf, \
         open(raw_path, "w", newline="") as rf:
        sw = csv.DictWriter(sf, fieldnames=sum_cols)
        sw.writeheader()
        rw = csv.DictWriter(rf, fieldnames=raw_cols)
        rw.writeheader()

        for alpha in tqdm(alphas, desc="  α", colour="cyan"):
            results = _run_parallel(G_sub, k, alpha, M, paths, n_runs, rebalance=rebalance)
            taus = [r[1] for r in results]

            for run_id, tau, _ in results:
                rw.writerow({"param_value": alpha, "n_nodes": n,
                             "run_id": run_id, "tau": tau})
            rf.flush()

            stats = _stats(taus)
            row = {"param_name": "alpha", "param_value": alpha,
                   "n_nodes": n, "n_edges": G_sub.number_of_edges()}
            row.update({k_: f"{v:.2f}" if isinstance(v, float) else v
                        for k_, v in stats.items()})
            sw.writerow(row)
            sf.flush()

            tqdm.write(f"    α={alpha:.2f} → τ={stats['tau_mean']:.0f} "
                       f"± {stats['tau_std']:.0f}")

    print(f"  ✓ {sum_path}")
    print(f"  ✓ {raw_path}")


# ═══════════════════════════════════════════════════════════════
#  EXP SN_B: τ vs k sulla LN reale
# ═══════════════════════════════════════════════════════════════

def run_exp_SN_B(force=False, top_n=60, rebalance=False):
    """
    Sottografo top-N della vera LN.
    α ∈ {0.0, 0.25, 0.5}, k variabile.
    """
    exp_name = "SN_B_tau_vs_k"
    sum_path = os.path.join(DATA_DIR, f"{exp_name}_summary.csv")
    raw_path = os.path.join(DATA_DIR, f"{exp_name}_raw.csv")
    slope_path = os.path.join(DATA_DIR, f"{exp_name}_slopes.csv")

    if os.path.exists(sum_path) and not force:
        print(f"  ⚠ {sum_path} esiste — usa --force. Salto.")
        return

    print(f"\n═══ EXP SN_B: τ vs k (LN Reale, top {top_n} nodi) ═══")

    G, alias_map = load_ln_snapshot()
    G_sub, node_info = extract_subgraph(G, alias_map, top_n=top_n)

    n = G_sub.number_of_nodes()
    n_merchants = max(1, int(math.sqrt(n)))
    M = list(range(n_merchants))
    alphas = [0.0, 0.25, 0.5]
    k_values = [10, 20, 50, 100, 200]
    n_runs = 12

    print(f"  n = {n}, M = {n_merchants} merchant (top hub)")

    paths = precompute_paths(G_sub)

    os.makedirs(DATA_DIR, exist_ok=True)

    sum_cols = ["param_name", "param_value", "n_nodes", "n_edges",
                "tau_mean", "tau_std", "tau_min", "tau_max", "tau_median",
                "ci95_low", "ci95_high", "n_runs", "alpha"]
    raw_cols = ["param_value", "n_nodes", "run_id", "tau", "alpha"]

    slope_data = []

    with open(sum_path, "w", newline="") as sf, \
         open(raw_path, "w", newline="") as rf:
        sw = csv.DictWriter(sf, fieldnames=sum_cols)
        sw.writeheader()
        rw = csv.DictWriter(rf, fieldnames=raw_cols)
        rw.writeheader()

        for alpha in alphas:
            print(f"\n  ── α = {alpha} ──")
            means_for_slope = []

            for k in tqdm(k_values, desc=f"  k (α={alpha})", colour="cyan"):
                results = _run_parallel(G_sub, k, alpha, M, paths, n_runs, rebalance=rebalance)
                taus = [r[1] for r in results]

                for run_id, tau, _ in results:
                    rw.writerow({"param_value": k, "n_nodes": n,
                                 "run_id": run_id, "tau": tau, "alpha": alpha})
                rf.flush()

                stats = _stats(taus)
                means_for_slope.append(stats["tau_mean"])
                row = {"param_name": "k", "param_value": k,
                       "n_nodes": n, "n_edges": G_sub.number_of_edges(),
                       "alpha": alpha}
                row.update({k_: f"{v:.2f}" if isinstance(v, float) else v
                            for k_, v in stats.items()})
                sw.writerow(row)
                sf.flush()

            log_k = np.log(k_values)
            log_tau = np.log(np.maximum(means_for_slope, 1e-9))
            slope = np.polyfit(log_k, log_tau, 1)[0]
            slope_data.append({"alpha": alpha, "n": n, "slope": slope})
            print(f"  Slope log-log (α={alpha}): {slope:.3f}")

    with open(slope_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "n", "slope"])
        w.writeheader()
        for dd in slope_data:
            dd_fmt = dd.copy()
            dd_fmt["slope"] = f"{dd['slope']:.4f}"
            w.writerow(dd_fmt)

    print(f"  ✓ {sum_path}")
    print(f"  ✓ {raw_path}")
    print(f"  ✓ {slope_path}")


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


def plot_SN_A():
    """τ vs α sulla LN reale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    sum_path = os.path.join(DATA_DIR, "SN_A_tau_vs_alpha_summary.csv")
    if not os.path.exists(sum_path):
        print("  ⚠ Dati SN_A non trovati, salto plot.")
        return

    summary = _load_csv(sum_path)
    alphas = [float(r["param_value"]) for r in summary]
    means = [float(r["tau_mean"]) for r in summary]
    ci_low = [float(r["ci95_low"]) for r in summary]
    ci_high = [float(r["ci95_high"]) for r in summary]
    n = int(summary[0]["n_nodes"])
    n_edges = int(summary[0]["n_edges"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(alphas, ci_low, ci_high, color="#e74c3c", alpha=0.15)
    ax.plot(alphas, means, color="#e74c3c", linewidth=2.5,
            marker="o", markersize=5, label=f"LN Reale (n={n}, e={n_edges})")

    ax.set_yscale("log")
    ax.set_title("Sink-Effect sulla Lightning Network Reale\n"
                 f"(Top {n} nodi per grado, k=50, M=[√n] hub come merchant)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Probabilità bias merchant α", fontsize=12)
    ax.set_ylabel("τ medio (scala logaritmica)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save(fig, "SN_A_tau_vs_alpha.png")


def plot_SN_B():
    """τ vs k log-log sulla LN reale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    sum_path = os.path.join(DATA_DIR, "SN_B_tau_vs_k_summary.csv")
    slope_path = os.path.join(DATA_DIR, "SN_B_tau_vs_k_slopes.csv")

    if not os.path.exists(sum_path):
        print("  ⚠ Dati SN_B non trovati, salto plot.")
        return

    summary = _load_csv(sum_path)
    data = {}
    for r in summary:
        a = float(r["alpha"])
        if a not in data:
            data[a] = {"k": [], "mean": []}
        data[a]["k"].append(float(r["param_value"]))
        data[a]["mean"].append(float(r["tau_mean"]))

    slopes = {}
    if os.path.exists(slope_path):
        for r in _load_csv(slope_path):
            slopes[float(r["alpha"])] = float(r["slope"])

    colors = {0.0: "#2ecc71", 0.25: "#e67e22", 0.5: "#e74c3c"}
    n = int(summary[0]["n_nodes"])

    fig, ax = plt.subplots(figsize=(9, 6))

    for alpha in sorted(data.keys()):
        d = data[alpha]
        k_arr = np.array(d["k"])
        m_arr = np.array(d["mean"])
        col = colors.get(alpha, "#95a5a6")
        sl = slopes.get(alpha, None)
        label = f"α={alpha}"
        if sl is not None:
            label += f" (slope={sl:.2f})"

        ax.plot(k_arr, m_arr, color=col, marker="o", markersize=6,
                linewidth=2.5, label=label)

        # Fit line
        if sl is not None:
            log_k = np.log(k_arr)
            log_m = np.log(np.maximum(m_arr, 1e-9))
            intercept = np.mean(log_m - sl * log_k)
            fit_tau = np.exp(intercept + sl * np.log(k_arr))
            ax.plot(k_arr, fit_tau, color=col, linestyle="--",
                    linewidth=1, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"$\\tau$ vs k nella Lightning Network\n"
                 f"(Top {n} nodi, slope ≈ 2 → ≈ 1 con merchant bias)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Capacità k (scala log)", fontsize=12)
    ax.set_ylabel("τ medio (scala log)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save(fig, "SN_B_tau_vs_k_loglog.png")


def plot_all_snapshot():
    print("\n═══ Generazione grafici LN Reale ═══")
    plot_SN_A()
    plot_SN_B()
    print("  Done!")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esperimenti su LN Reale")
    parser.add_argument("--exp", choices=["SN_A", "SN_B", "all"], default="all")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Solo analisi del grafo, nessuna simulazione")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-n", type=int, default=60,
                        help="Numero di top nodi da estrarre (default 60)")
    parser.add_argument("--rebalance", action="store_true",
                        help="Abilita il JIT Circular Rebalancing intelligente")
    parser.add_argument("--suffix", type=str, default="",
                        help="Aggiunge un suffisso alle cartelle di output per non sovrascrivere")

    args = parser.parse_args()
    
    # Previene il mescolamento dei dati del nuovo algoritmo con le simulazioni standard/vecchie
    if args.rebalance and not args.suffix:
        args.suffix = "blind_probing"
        print(f"\n  [INFO] Auto-impostato suffix='{args.suffix}' per separare i dati del rebalancing privacy-preserving.")
        
    set_output_dirs(args.suffix)

    if args.analyze_only:
        analyze_snapshot(top_n=args.top_n)
    elif args.plot_only:
        plot_all_snapshot()
    else:
        dispatch = {"SN_A": lambda f, r: run_exp_SN_A(f, args.top_n, r),
                    "SN_B": lambda f, r: run_exp_SN_B(f, args.top_n, r)}
        exps = ["SN_A", "SN_B"] if args.exp == "all" else [args.exp]

        t0 = time.time()
        for e in exps:
            dispatch[e](args.force, args.rebalance)
        elapsed = time.time() - t0
        print(f"\n  Completato in {elapsed:.0f}s")

        plot_all_snapshot()
