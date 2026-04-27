"""
runner.py — Esecuzione degli esperimenti e salvataggio dati
=============================================================
Motore logistico che orchestra l'esecuzione massiva delle simulazioni.
Implementa 4 esperimenti formali analitici:
- A (tau_vs_alpha) : Declino della vita della rete al variare del bias α.
- B (tau_vs_k) : Analisi delle leggi di scala spaziali (O(k²) vs O(k)).
- C (dumbbell_scatter) : Studio microscopico dello spostamento del collo di bottiglia topologico.
- D (transizione_alpha) : Calcolo del punto critico di transizione di fase (α*).

NOTA TECNICA SUL PARALLELISMO:
Utilizza `multiprocessing.Pool` con il pattern "initializer" (variabili globali pre-biforcazione)
per eludere il costoso e fallibile pickling (serializzazione) dei macro-oggetti `Graph` e `Paths`
che causerebbe crash o pesanti rallentamenti nativi su architetture macOS ("spawn" start method).
Ogni processo figlio riceve unicamente due interi leggeri: `(seed, run_id)`.
"""

import csv
import math
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from .core import precompute_paths, simulate
from .graphs import (
    make_clique, make_dumbbell,
    get_edge_betweenness, classify_edge,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_multi")


# ─────────────────────────────────────────────────
#  Globals per il pool di worker (evita pickling)
# ─────────────────────────────────────────────────

_G = None
_k = None
_alpha = None
_M = None
_paths = None


def _init_pool(G, k, alpha, M, paths):
    """
    Funzione di Inizializzazione per Processo Worker (Pool Initializer).
    Questa funzione viene eseguita esattamente una volta da ciascun processo
    'figlio' non appena viene generato dalla CPU, clonando le variabili
    pesanti nello scope globale del figlio così da non doverle inviare
    tramite tubo interprocesso per ogni singola transazione simulata.
    """
    global _G, _k, _alpha, _M, _paths
    _G = G
    _k = k
    _alpha = alpha
    _M = M
    _paths = paths


def _worker(args):
    """
    Micro-task isolato del processo worker.
    Essendo il contesto già caricato in memoria tramite l'initializer,
    riceve soltanto la matrice di randomizzazione (seed) e l'identificativo.
    """
    seed, run_id = args
    tau, failed_edge = simulate(_G, _k, _alpha, _M, _paths, seed=seed)
    return run_id, tau, failed_edge


# ─────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _summary_path(exp_name):
    return os.path.join(DATA_DIR, f"{exp_name}_summary.csv")


def _raw_path(exp_name):
    return os.path.join(DATA_DIR, f"{exp_name}_raw.csv")


def _should_run(exp_name, force):
    p = _summary_path(exp_name)
    if os.path.exists(p) and not force:
        print(f"  ⚠ {p} esiste già — usa --force per sovrascrivere. Salto.")
        return False
    return True


def _run_parallel(G, k, alpha, M, paths, n_runs):
    """
    Esegue un batch di 'n_runs' simulazioni in parallelo sfruttando 
    tutti i core logici disponibili sul sistema host. Restituisce i
    risultati disordinati per massimizzare il throughput, poi li riordina.
    """
    # Genera un seed univoco predicibile per ogni run (1000 + i)
    tasks = [(1000 + i, i) for i in range(n_runs)]
    with Pool(
        processes=cpu_count(),
        initializer=_init_pool,
        initargs=(G, k, alpha, M, paths),
    ) as pool:
        # imap_unordered è più veloce del map standard in multiprocessing asincrono
        results = list(tqdm(
            pool.imap_unordered(_worker, tasks),
            total=n_runs,
            desc="    runs",
            leave=False,
            colour="green",
        ))
    # Riordina per run_id crescente per correttezza log e compatibilità
    results.sort(key=lambda x: x[0])
    return results


def _compute_stats(taus):
    """
    Calcola rapidamente media empirica, deviazione standard campionaria (ddof=1)
    e Intervallo di Confidenza Assoluto (CI) al 95% sfruttando distribuzioni asintotiche.
    """
    arr = np.array(taus, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n) # Standard Error
    return {
        "tau_mean": mean,
        "tau_std": std,
        "tau_min": np.min(arr),
        "tau_max": np.max(arr),
        "tau_median": np.median(arr),
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "n_runs": n,
    }


# ═══════════════════════════════════════════════════════════════
#  EXP A: tau_vs_alpha
# ═══════════════════════════════════════════════════════════════

def run_exp_A(force=False):
    """
    Clique K_n, k=50, M=[0..sqrt(n)-1], alpha variabile.
    n in [10, 20, 30, 40, 50, 60]. 150 run per config.
    """
    exp_name = "A_tau_vs_alpha"
    sum_path = _summary_path(exp_name)
    raw_path = _raw_path(exp_name)

    if os.path.exists(sum_path) and not force:
        print(f"  ⚠ {sum_path} esiste. Riprendo le simulazioni mancanti (Append Mode).")
    elif force:
        if os.path.exists(sum_path): os.remove(sum_path)
        if os.path.exists(raw_path): os.remove(raw_path)

    print("\n═══ EXP A: τ vs α (Finite Size Scaling) ═══")

    n_values = [10, 40, 80, 120, 160, 200]
    k = 50
    alphas = [round(a * 0.05, 2) for a in range(13)]  # 0.00..0.60
    n_runs = 12

    os.makedirs(DATA_DIR, exist_ok=True)

    sum_cols = ["param_name", "param_value", "n_nodes", "tau_mean", "tau_std",
                "tau_min", "tau_max", "tau_median",
                "ci95_low", "ci95_high", "n_runs"]
    raw_cols = ["param_value", "n_nodes", "run_id", "tau"]

    completed_points = set()
    if os.path.exists(sum_path):
        with open(sum_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_points.add((int(r["n_nodes"]), float(r["param_value"])))

    write_headers = not os.path.exists(sum_path)

    with open(sum_path, "a", newline="") as sf, \
         open(raw_path, "a", newline="") as rf:
        sw = csv.DictWriter(sf, fieldnames=sum_cols)
        if write_headers: sw.writeheader()
        rw = csv.DictWriter(rf, fieldnames=raw_cols)
        if write_headers: rw.writeheader()

        for n in n_values:
            print(f"\n  ── n = {n} ──")
            G = make_clique(n)
            M = list(range(max(1, int(math.sqrt(n)))))
            paths = precompute_paths(G)
            
            for alpha in tqdm(alphas, desc=f"  α (n={n})", colour="cyan"):
                if (n, alpha) in completed_points:
                    tqdm.write(f"  [SKIPPED] n={n}, α={alpha}")
                    continue
                results = _run_parallel(G, k, alpha, M, paths, n_runs)
                taus = [r[1] for r in results]

                # Raw
                for run_id, tau, _ in results:
                    rw.writerow({"param_value": alpha, "n_nodes": n, "run_id": run_id, "tau": tau})
                rf.flush()

                # Summary
                stats = _compute_stats(taus)
                row = {"param_name": "alpha", "param_value": alpha, "n_nodes": n}
                row.update({k_: f"{v:.2f}" if isinstance(v, float) else v
                            for k_, v in stats.items()})
                sw.writerow(row)
                sf.flush()

                tqdm.write(f"    α={alpha:.2f} (n={n}) → τ={stats['tau_mean']:.0f} "
                           f"± {stats['tau_std']:.0f}")

    print(f"  ✓ Salvato: {sum_path}")
    print(f"  ✓ Salvato: {raw_path}")


# ═══════════════════════════════════════════════════════════════
#  EXP B: tau_vs_k
# ═══════════════════════════════════════════════════════════════

def run_exp_B(force=False):
    """
    Clique K_n, M=[0..sqrt(n)-1], alpha ∈ {0.0, 0.5}, k variabile.
    n in [10, 30, 60] per mostrare le slope differenti.
    150 run per.
    """
    exp_name = "B_tau_vs_k"

    print("\n═══ EXP B: τ vs k (log-log FSS) ═══")

    n_values = [100]
    alphas = [0.0, 0.25, 0.5]
    k_values = [10, 20, 50, 100, 200, 400, 600, 800, 1000]
    n_runs = 12

    os.makedirs(DATA_DIR, exist_ok=True)
    sum_path = _summary_path(exp_name)
    raw_path = _raw_path(exp_name)
    slope_path = os.path.join(DATA_DIR, f"{exp_name}_slopes.csv")

    if os.path.exists(sum_path) and not force:
        print(f"  ⚠ {sum_path} esiste. Riprendo (Append Mode).")
    elif force:
        if os.path.exists(sum_path): os.remove(sum_path)
        if os.path.exists(raw_path): os.remove(raw_path)

    sum_cols = ["param_name", "param_value", "n_nodes", "tau_mean", "tau_std",
                "tau_min", "tau_max", "tau_median",
                "ci95_low", "ci95_high", "n_runs", "alpha"]
    raw_cols = ["param_value", "n_nodes", "run_id", "tau", "alpha"]

    slope_data = []

    completed_points = {}
    if os.path.exists(sum_path):
        with open(sum_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_points[(float(r["alpha"]), int(r["n_nodes"]), int(r["param_value"]))] = float(r["tau_mean"])

    write_headers = not os.path.exists(sum_path)

    with open(sum_path, "a", newline="") as sf, \
         open(raw_path, "a", newline="") as rf:
        sw = csv.DictWriter(sf, fieldnames=sum_cols)
        if write_headers: sw.writeheader()
        rw = csv.DictWriter(rf, fieldnames=raw_cols)
        if write_headers: rw.writeheader()

        for alpha in alphas:
            print(f"\n  ── α = {alpha} ──")
            for n in n_values:
                G = make_clique(n)
                M = list(range(max(1, int(math.sqrt(n)))))
                paths = precompute_paths(G)
                means_for_slope = []

                for k in tqdm(k_values, desc=f"  k (α={alpha}, n={n})", colour="cyan"):
                    if (alpha, n, k) in completed_points:
                        tqdm.write(f"  [SKIPPED] α={alpha}, n={n}, k={k}")
                        means_for_slope.append(completed_points[(alpha, n, k)])
                        continue
                        
                    results = _run_parallel(G, k, alpha, M, paths, n_runs)
                    taus = [r[1] for r in results]

                    for run_id, tau, _ in results:
                        rw.writerow({"param_value": k, "n_nodes": n, "run_id": run_id,
                                     "tau": tau, "alpha": alpha})
                    rf.flush()

                    stats = _compute_stats(taus)
                    means_for_slope.append(stats["tau_mean"])
                    row = {"param_name": "k", "param_value": k, "n_nodes": n, "alpha": alpha}
                    row.update({k_: f"{v:.2f}" if isinstance(v, float) else v
                                for k_, v in stats.items()})
                    sw.writerow(row)
                    sf.flush()

                log_k = np.log(k_values)
                log_tau = np.log(np.maximum(means_for_slope, 1e-9))
                slope = np.polyfit(log_k, log_tau, 1)[0]
                slope_data.append({"alpha": alpha, "n": n, "slope": slope})
                print(f"  Slope log-log (α={alpha}, n={n}): {slope:.3f}")

    with open(slope_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "n", "slope"])
        w.writeheader()
        for d in slope_data:
            d_fmt = d.copy()
            d_fmt["slope"] = f"{d['slope']:.4f}"
            w.writerow(d_fmt)

    print(f"  ✓ Salvato: {sum_path}")
    print(f"  ✓ Salvato: {raw_path}")
    print(f"  ✓ Salvato: {slope_path}")


# ═══════════════════════════════════════════════════════════════
#  EXP C: dumbbell_scatter
# ═══════════════════════════════════════════════════════════════

def run_exp_C(force=False):
    """
    Dumbbell n in [10, 20, 30, 40, 50, 60] (c=n/2). k=100.
    150 run per config per limitare i tempi.
    """
    exp_name = "C_dumbbell_scatter"
    if not _should_run(exp_name, force):
        return

    print("\n═══ EXP C: Dumbbell scatter (FSS) ═══")

    n_values = [10, 50, 100, 150, 200]
    k = 100
    alphas = [0.0, 0.3, 0.6]
    n_runs = 12

    os.makedirs(DATA_DIR, exist_ok=True)
    sum_path = _summary_path(exp_name)
    raw_path = _raw_path(exp_name)
    freq_path = os.path.join(DATA_DIR, f"{exp_name}_fail_freq.csv")

    sum_cols = ["param_name", "param_value", "n_nodes", "tau_mean", "tau_std",
                "tau_min", "tau_max", "tau_median",
                "ci95_low", "ci95_high", "n_runs"]
    raw_cols = ["param_value", "n_nodes", "run_id", "tau", "failed_edge_u", "failed_edge_v"]
    freq_cols = ["alpha", "n_nodes", "edge_u", "edge_v", "betweenness",
                 "fail_count", "fail_freq", "edge_type"]

    with open(sum_path, "w", newline="") as sf, \
         open(raw_path, "w", newline="") as rf, \
         open(freq_path, "w", newline="") as ff:
        sw = csv.DictWriter(sf, fieldnames=sum_cols)
        sw.writeheader()
        rw = csv.DictWriter(rf, fieldnames=raw_cols)
        rw.writeheader()
        fw = csv.DictWriter(ff, fieldnames=freq_cols)
        fw.writeheader()

        for n in n_values:
            c = n // 2
            G = make_dumbbell(c)
            # Metà merchant su cricca A, metà su cricca B
            nm = max(1, int(math.sqrt(n)))
            M = []
            for i in range(nm):
                if i % 2 == 0:
                    M.append(i // 2)
                else:
                    M.append(c + (i // 2))
            
            bridge_edge = (c - 1, c)
            paths = precompute_paths(G)
            eb = get_edge_betweenness(G)
            M_set = set(M)

            for alpha in tqdm(alphas, desc=f"  α (n={n})", colour="cyan"):
                results = _run_parallel(G, k, alpha, M, paths, n_runs)
                taus = [r[1] for r in results]

                for run_id, tau, fe in results:
                    rw.writerow({
                        "param_value": alpha, "n_nodes": n, "run_id": run_id, "tau": tau,
                        "failed_edge_u": fe[0], "failed_edge_v": fe[1],
                    })
                rf.flush()

                stats = _compute_stats(taus)
                row = {"param_name": "alpha", "param_value": alpha, "n_nodes": n}
                row.update({k_: f"{v:.2f}" if isinstance(v, float) else v
                            for k_, v in stats.items()})
                sw.writerow(row)
                sf.flush()

                fail_counts = {}
                for _, _, fe in results:
                    fail_counts[fe] = fail_counts.get(fe, 0) + 1

                all_edges = sorted(eb.keys())
                for edge in all_edges:
                    cnt = fail_counts.get(edge, 0)
                    etype = classify_edge(edge[0], edge[1], M_set, bridge_edge)
                    fw.writerow({
                        "alpha": alpha, "n_nodes": n,
                        "edge_u": edge[0], "edge_v": edge[1],
                        "betweenness": f"{eb[edge]:.1f}",
                        "fail_count": cnt,
                        "fail_freq": f"{cnt / n_runs:.4f}",
                        "edge_type": etype,
                    })
                ff.flush()

                tqdm.write(f"    α={alpha:.1f} (n={n}) → τ={stats['tau_mean']:.0f} "
                           f"± {stats['tau_std']:.0f}")

    print(f"  ✓ Salvato: {sum_path}")
    print(f"  ✓ Salvato: {raw_path}")
    print(f"  ✓ Salvato: {freq_path}")


# ═══════════════════════════════════════════════════════════════
#  EXP D: transizione_alpha
# ═══════════════════════════════════════════════════════════════

def _wilson_ci(x, n, z=1.96):
    """Intervallo di confidenza Wilson per una proporzione."""
    if n == 0:
        return 0.0, 0.0
    p_hat = x / n
    center = (x + z**2 / 2) / (n + z**2)
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) \
           / (1 + z**2 / n)
    return max(0.0, center - half), min(1.0, center + half)


def run_exp_D(force=False):
    """
    Dumbbell n in [10, 20, 30, 40, 50, 60], alpha da 0 a 1.
    150 run per config ed estrazione di alpha* in funzione di n.
    """
    exp_name = "D_transizione_alpha"
    if not _should_run(exp_name, force):
        return

    print("\n═══ EXP D: Transizione α* vs n (FSS) ═══")

    n_values = [10, 50, 100, 150, 200]
    k = 100
    alphas = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    n_runs = 12

    os.makedirs(DATA_DIR, exist_ok=True)
    sum_path = _summary_path(exp_name)
    raw_path = _raw_path(exp_name)
    trans_path = os.path.join(DATA_DIR, f"{exp_name}_transition.csv")
    star_path = os.path.join(DATA_DIR, f"{exp_name}_alpha_star_vs_n.csv")

    sum_cols = ["param_name", "param_value", "n_nodes", "tau_mean", "tau_std",
                "tau_min", "tau_max", "tau_median",
                "ci95_low", "ci95_high", "n_runs"]
    raw_cols = ["param_value", "n_nodes", "run_id", "tau", "fail_type"]
    trans_cols = ["alpha", "n_nodes", "p_merchant", "p_bridge", "p_other",
                  "ci_low_merchant", "ci_high_merchant"]

    alpha_stars = []

    with open(sum_path, "w", newline="") as sf, \
         open(raw_path, "w", newline="") as rf, \
         open(trans_path, "w", newline="") as tf:
        sw = csv.DictWriter(sf, fieldnames=sum_cols)
        sw.writeheader()
        rw = csv.DictWriter(rf, fieldnames=raw_cols)
        rw.writeheader()
        tw = csv.DictWriter(tf, fieldnames=trans_cols)
        tw.writeheader()

        for n in n_values:
            print(f"\n  ── n = {n} ──")
            c = n // 2
            G = make_dumbbell(c)
            nm = max(1, int(math.sqrt(n)))
            M = []
            for i in range(nm):
                if i % 2 == 0:
                    M.append(i // 2)
                else:
                    M.append(c + (i // 2))

            bridge_edge = (c - 1, c)
            M_set = set(M)
            paths = precompute_paths(G)
            
            p_merchant_list = []

            for alpha in tqdm(alphas, desc=f"  α (n={n})", colour="cyan"):
                results = _run_parallel(G, k, alpha, M, paths, n_runs)
                taus = [r[1] for r in results]

                counts = {"merchant_adj": 0, "bridge": 0, "other": 0}
                for run_id, tau, fe in results:
                    etype = classify_edge(fe[0], fe[1], M_set, bridge_edge)
                    counts[etype] += 1
                    rw.writerow({
                        "param_value": alpha, "n_nodes": n, "run_id": run_id,
                        "tau": tau, "fail_type": etype,
                    })
                rf.flush()

                stats = _compute_stats(taus)
                row = {"param_name": "alpha", "param_value": alpha, "n_nodes": n}
                row.update({k_: f"{v:.2f}" if isinstance(v, float) else v
                            for k_, v in stats.items()})
                sw.writerow(row)
                sf.flush()

                p_m = counts["merchant_adj"] / n_runs
                p_b = counts["bridge"] / n_runs
                p_o = counts["other"] / n_runs
                ci_low, ci_high = _wilson_ci(counts["merchant_adj"], n_runs)

                p_merchant_list.append((alpha, p_m))

                tw.writerow({
                    "alpha": alpha, "n_nodes": n,
                    "p_merchant": f"{p_m:.4f}",
                    "p_bridge": f"{p_b:.4f}",
                    "p_other": f"{p_o:.4f}",
                    "ci_low_merchant": f"{ci_low:.4f}",
                    "ci_high_merchant": f"{ci_high:.4f}",
                })
                tf.flush()

            astar = _estimate_alpha_star(p_merchant_list)
            alpha_stars.append({"n": n, "alpha_star": astar})
            print(f"  α* stimato (n={n}) = {astar:.4f}")

    with open(star_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "alpha_star"])
        w.writeheader()
        for a in alpha_stars:
            w.writerow({"n": a["n"], "alpha_star": f"{a['alpha_star']:.4f}"})

    print(f"  ✓ Salvato: {sum_path}")
    print(f"  ✓ Salvato: {raw_path}")
    print(f"  ✓ Salvato: {trans_path}")
    print(f"  ✓ Salvato: {star_path}")


def _estimate_alpha_star(p_list):
    """
    Interpolazione lineare tra i due valori di alpha più vicini
    a p_merchant = 0.5.
    """
    for i in range(len(p_list) - 1):
        a1, p1 = p_list[i]
        a2, p2 = p_list[i + 1]
        if (p1 - 0.5) * (p2 - 0.5) <= 0:
            if abs(p2 - p1) < 1e-12:
                return (a1 + a2) / 2
            return a1 + (0.5 - p1) * (a2 - a1) / (p2 - p1)
    best = min(p_list, key=lambda x: abs(x[1] - 0.5))
    return best[0]
