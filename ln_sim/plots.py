"""
plots.py — Generazione grafici per la tesi
============================================
5 grafici: A (τ vs α), B (τ vs k log-log), C (dumbbell scatter),
D (transizione α*), E (confronto α=0 vs α=0.6).
"""

import csv
import os

from .graphs import make_dumbbell, get_edge_betweenness

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots_multi")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_multi")

# ── Stile globale ──
plt.style.use("seaborn-v0_8-whitegrid")
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
DPI = 300


def _load_csv(path):
    """Carica CSV e restituisce lista di dict."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════
#  GRAFICO A — τ vs α (FSS)
# ═══════════════════════════════════════════════════════════════

def plot_A():
    """τ vs α con scatter raw e CI 95% per vari n."""
    sum_path = os.path.join(DATA_DIR, "A_tau_vs_alpha_summary.csv")

    if not os.path.exists(sum_path):
        print("  ⚠ Dati EXP A non trovati, salto plot A.")
        return

    summary = _load_csv(sum_path)
    # Raggruppa per n
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

    n_sorted = sorted(by_n.keys())
    cmap = plt.get_cmap("viridis")
    
    for i, n in enumerate(n_sorted):
        color = cmap(i / max(1, len(n_sorted) - 1))
        d = by_n[n]
        # CI band (opzionale se troppe righe, ma teniamolo molto trasparente)
        ax.fill_between(d["alphas"], d["ci_low"], d["ci_high"],
                        color=color, alpha=0.1, zorder=2)
        ax.plot(d["alphas"], d["means"], color=color, linewidth=2,
                marker="o", markersize=4, zorder=3, label=f"n={n}")

    # Reference line
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1,
               label="α=0: uniforme")

    ax.set_yscale("log")
    ax.set_title("Tempo di fallimento τ al variare del bias α\n"
                 "(Finite Size Scaling: Clique n nodi, k=50, M=[√n] merchant)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Probabilità bias merchant α", fontsize=LABEL_SIZE)
    ax.set_ylabel("τ medio (scala logaritmica)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, title="Dimensione Rete")
    fig.tight_layout()
    _save(fig, "A_tau_vs_alpha.png")


# ═══════════════════════════════════════════════════════════════
#  GRAFICO B — τ vs k (log-log FSS)
# ═══════════════════════════════════════════════════════════════

def plot_B():
    """τ vs k in scala log-log — solo il grafo più grande per chiarezza."""
    sum_path = os.path.join(DATA_DIR, "B_tau_vs_k_summary.csv")
    slope_path = os.path.join(DATA_DIR, "B_tau_vs_k_slopes.csv")

    if not os.path.exists(sum_path):
        print("  ⚠ Dati EXP B non trovati, salto plot B.")
        return

    summary = _load_csv(sum_path)

    # Trova il N massimo disponibile
    max_n = max(int(r["n_nodes"]) for r in summary)

    # Filtra solo il grafo più grande e raggruppa per alpha
    data = {}
    for r in summary:
        n = int(r["n_nodes"])
        if n != max_n:
            continue
        a = float(r["alpha"])
        if a not in data:
            data[a] = {"k": [], "mean": [], "ci_low": [], "ci_high": []}
        data[a]["k"].append(float(r["param_value"]))
        data[a]["mean"].append(float(r["tau_mean"]))
        data[a]["ci_low"].append(float(r["ci95_low"]))
        data[a]["ci_high"].append(float(r["ci95_high"]))

    # Carica slopes (solo per max_n)
    slopes = {}
    if os.path.exists(slope_path):
        for r in _load_csv(slope_path):
            if int(r["n"]) == max_n:
                slopes[float(r["alpha"])] = float(r["slope"])

    # Palette distinta per 3 alpha
    colors = {0.0: "#2980b9", 0.25: "#27ae60", 0.5: "#e74c3c"}
    markers = {0.0: "o", 0.25: "s", 0.5: "D"}
    labels = {
        0.0:  "α = 0 (uniforme)",
        0.25: "α = 0.25",
        0.5:  "α = 0.5",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for alpha in sorted(data.keys()):
        d = data[alpha]
        k_arr = np.array(d["k"])
        m_arr = np.array(d["mean"])
        col = colors.get(alpha, "#7f8c8d")
        mk = markers.get(alpha, "o")

        # CI band
        ax.fill_between(k_arr, d["ci_low"], d["ci_high"],
                         color=col, alpha=0.12)

        # Dati  
        ax.plot(k_arr, m_arr, color=col, marker=mk, markersize=7,
                linewidth=2.5, label=labels.get(alpha, f"α={alpha}"), zorder=3)

        # Fit power-law tratteggiato
        if alpha in slopes:
            s = slopes[alpha]
            log_k = np.log(k_arr)
            log_m = np.log(np.maximum(m_arr, 1e-9))
            intercept = np.mean(log_m - s * log_k)
            k_dense = np.logspace(np.log10(k_arr.min()), np.log10(k_arr.max()), 50)
            fit_tau = np.exp(intercept + s * np.log(k_dense))
            ax.plot(k_dense, fit_tau, color=col, linestyle="--",
                    linewidth=1.5, alpha=0.6,
                    label=f"  fit: slope = {s:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Scaling di τ con la capacità k  (Clique n={max_n})\n",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Capacità k (scala log)", fontsize=LABEL_SIZE)
    ax.set_ylabel("τ medio (scala log)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left")
    fig.tight_layout()
    _save(fig, "B_tau_vs_k_loglog.png")


# ═══════════════════════════════════════════════════════════════
#  GRAFICO C — Dumbbell scatter (Seleziona il max N)
# ═══════════════════════════════════════════════════════════════

def plot_C():
    """Tre pannelli: scatter betw vs fail freq per alpha. Plotta solo il MAX N."""
    freq_path = os.path.join(DATA_DIR, "C_dumbbell_scatter_fail_freq.csv")

    if not os.path.exists(freq_path):
        print("  ⚠ Dati EXP C non trovati, salto plot C.")
        return

    rows = _load_csv(freq_path)
    if not rows:
        return
        
    # Trova il N massimo
    max_n = max(int(r["n_nodes"]) for r in rows)
    rows = [r for r in rows if int(r["n_nodes"]) == max_n]

    # Raggruppa per alpha
    by_alpha = {}
    for r in rows:
        a = float(r["alpha"])
        if a not in by_alpha:
            by_alpha[a] = []
        by_alpha[a].append(r)

    alphas_sorted = sorted(by_alpha.keys())
    n_panels = len(alphas_sorted)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    style = {
        "merchant_adj": {"color": "#e74c3c", "marker": "o", "s": 120,
                         "label": "archi merchant"},
        "bridge":       {"color": "#2980b9", "marker": "D", "s": 150,
                         "label": "ponte"},
        "other":        {"color": "#27ae60", "marker": "s", "s": 80,
                         "label": "altri archi"},
    }

    for ax, alpha in zip(axes, alphas_sorted):
        data = by_alpha[alpha]

        for r in data:
            etype = r["edge_type"]
            bet = float(r["betweenness"])
            freq = float(r["fail_freq"])
            st = style[etype]
            ax.scatter(bet, freq, c=st["color"], marker=st["marker"],
                       s=st["s"], zorder=3, edgecolors="white", linewidth=0.5)

            # Annotazione ponte
            if etype == "bridge":
                ax.annotate(f"ponte\ng(e)={bet:.0f}",
                            xy=(bet, freq),
                            xytext=(bet + 2, freq + 0.08),
                            fontsize=8,
                            arrowprops=dict(arrowstyle="->", color="#2980b9"),
                            color="#2980b9")

        ax.set_title(f"α = {alpha}", fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_xlabel("Betweenness g(e)", fontsize=LABEL_SIZE)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=TICK_SIZE)

    axes[0].set_ylabel("Frequenza primo fallimento", fontsize=LABEL_SIZE)

    # Legend (solo sul primo pannello)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=style[t]["marker"], color="w",
               markerfacecolor=style[t]["color"],
               markersize=10, label=style[t]["label"])
        for t in ["merchant_adj", "bridge", "other"]
    ]
    axes[0].legend(handles=handles, fontsize=LEGEND_SIZE - 1, loc="upper left")

    fig.suptitle(f"Arco critico nel grafo dumbbell al variare di α (n={max_n})",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "C_dumbbell_scatter.png")


# ═══════════════════════════════════════════════════════════════
#  GRAFICO D — Transizione α* vs n
# ═══════════════════════════════════════════════════════════════

def plot_D():
    """Pannello 1: Crossover su MAX N. Pannello 2: α* vs n."""
    trans_path = os.path.join(DATA_DIR, "D_transizione_alpha_transition.csv")
    star_path = os.path.join(DATA_DIR, "D_transizione_alpha_alpha_star_vs_n.csv")

    if not os.path.exists(trans_path):
        print("  ⚠ Dati EXP D non trovati, salto plot D.")
        return

    rows = _load_csv(trans_path)
    if not rows: return
    max_n = max(int(r["n_nodes"]) for r in rows)
    
    rows_max = [r for r in rows if int(r["n_nodes"]) == max_n]
    alphas = [float(r["alpha"]) for r in rows_max]
    p_m = [float(r["p_merchant"]) for r in rows_max]
    p_b = [float(r["p_bridge"]) for r in rows_max]

    alpha_stars = []
    if os.path.exists(star_path):
        star_rows = _load_csv(star_path)
        alpha_stars = [(int(r["n"]), float(r["alpha_star"])) for r in star_rows]
        
    alpha_stars.sort()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- PANNELLO 1: Transizione a N MAX ---
    ax = axes[0]
    ax.plot(alphas, p_m, color="#e74c3c", linewidth=2.5, marker="o", markersize=6, 
            label="P(sink MERCHANT)")
    ax.plot(alphas, p_b, color="#2980b9", linewidth=2.5, marker="D", markersize=6, 
            linestyle="--", alpha=0.8,
            label="P(sink PONTE)")
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.5,
               label="P(m) = P(b)")

    astar_max = next((val for n, val in alpha_stars if n == max_n), None)
    if astar_max:
        ax.axvline(x=astar_max, color="black", linestyle="-.",
                   linewidth=1.5, label=f"α* Empirico ≈ {astar_max:.3f}")
                   
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Visualizzazione Crossover Centrale (n={max_n})", fontsize=TITLE_SIZE)
    ax.set_xlabel("Bias merchant α", fontsize=LABEL_SIZE)
    ax.set_ylabel("Probabilità di esaurimento", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)

    # --- PANNELLO 2: α* vs n (Finite Size Scaling) ---
    ax2 = axes[1]
    if alpha_stars:
        ns = [x[0] for x in alpha_stars]
        asts = [x[1] for x in alpha_stars]
        ax2.plot(ns, asts, color="purple", marker="*", markersize=12, linewidth=2.5)
        ax2.set_title("Finite Size Scaling: α* vs n", fontsize=TITLE_SIZE)
        ax2.set_xlabel("Dimensione del network n", fontsize=LABEL_SIZE)
        ax2.set_ylabel("Soglia di transizione critica α*", fontsize=LABEL_SIZE)
        
        # Aggiunta linea teorica approssimativa di decay
        # Per un dumbbell puro asintotico, P(merchant) cresce, alpha* scala come ~ 1/n
        n_dense = np.linspace(min(ns), max(ns), 100)
        c_fit = asts[-1] * max_n
        ax2.plot(n_dense, c_fit/n_dense, '--', color="gray", label=f"Fit ~ 1/n")
        ax2.legend()

    fig.suptitle("Analisi della Transizione di Fase P(Merchant) vs P(Ponte)", fontweight="bold", fontsize=TITLE_SIZE+2)
    fig.tight_layout()
    _save(fig, "D_transizione_alpha.png")


# ═══════════════════════════════════════════════════════════════
#  GRAFICO E — Confronto α=0 vs α=0.6
# ═══════════════════════════════════════════════════════════════

def plot_E():
    """Due scatter dumbbell affiancati (α=0 e α=0.6) con frecce sul MAX N."""
    freq_path = os.path.join(DATA_DIR, "C_dumbbell_scatter_fail_freq.csv")

    if not os.path.exists(freq_path):
        print("  ⚠ Dati EXP C non trovati, salto plot E.")
        return

    rows = _load_csv(freq_path)
    if not rows: return
    max_n = max(int(r["n_nodes"]) for r in rows)
    rows = [r for r in rows if int(r["n_nodes"]) == max_n]

    by_alpha = {}
    for r in rows:
        a = float(r["alpha"])
        if a not in by_alpha:
            by_alpha[a] = []
        by_alpha[a].append(r)

    target_alphas = [0.0, 0.6]
    for a in target_alphas:
        if a not in by_alpha:
            print(f"  ⚠ α={a} non trovato nei dati, salto plot E.")
            return

    style = {
        "merchant_adj": {"color": "#e74c3c", "marker": "o", "s": 120,
                         "label": "archi merchant"},
        "bridge":       {"color": "#2980b9", "marker": "D", "s": 150,
                         "label": "ponte"},
        "other":        {"color": "#27ae60", "marker": "s", "s": 80,
                         "label": "altri archi"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, alpha in enumerate(target_alphas):
        ax = axes[ax_idx]
        data = by_alpha[alpha]

        for r in data:
            etype = r["edge_type"]
            bet = float(r["betweenness"])
            freq = float(r["fail_freq"])
            st = style[etype]
            ax.scatter(bet, freq, c=st["color"], marker=st["marker"],
                       s=st["s"], zorder=3, edgecolors="white", linewidth=0.5)

            if etype == "bridge":
                ax.annotate(f"ponte\n{freq:.2f}",
                            xy=(bet, freq),
                            xytext=(bet + 2, min(freq + 0.1, 0.9)),
                            fontsize=8,
                            arrowprops=dict(arrowstyle="->", color="#2980b9"),
                            color="#2980b9")

        subtitle = f"Prima del merchant (α={alpha})" if alpha == 0.0 \
                   else f"Con merchant attivo (α={alpha})"
        ax.set_title(subtitle, fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_xlabel("Betweenness g(e)", fontsize=LABEL_SIZE)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=TICK_SIZE)

    axes[0].set_ylabel("Frequenza primo fallimento", fontsize=LABEL_SIZE)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=style[t]["marker"], color="w",
               markerfacecolor=style[t]["color"],
               markersize=10, label=style[t]["label"])
        for t in ["merchant_adj", "bridge", "other"]
    ]
    axes[0].legend(handles=handles, fontsize=LEGEND_SIZE - 1, loc="upper left")

    fig.suptitle(f"Spostamento del collo di bottiglia (Grafo n={max_n})",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "E_confronto_alpha0_alpha06.png")


# ═══════════════════════════════════════════════════════════════
#  Genera tutti
# ═══════════════════════════════════════════════════════════════

def generate_all_plots():
    """Genera tutti i grafici dai CSV esistenti."""
    print("\n═══ Generazione grafici ═══")
    plot_A()
    plot_B()
    plot_C()
    plot_D()
    plot_E()
    print("  Done!")
