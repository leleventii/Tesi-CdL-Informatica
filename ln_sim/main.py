#!/usr/bin/env python3
"""
main.py — Entry point per il simulatore LN
============================================
Uso:
    python -m ln_sim.main              # esegui tutti gli esperimenti + grafici
    python -m ln_sim.main --exp A      # solo esperimento A
    python -m ln_sim.main --plot-only  # genera grafici dai CSV esistenti
    python -m ln_sim.main --force      # sovrascrivi CSV esistenti
"""

import argparse
import time

from .runners.runner import run_exp_A, run_exp_B, run_exp_C, run_exp_D
from .visualization.plots import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Simulatore LN — Tesi Ventilii 2025"
    )
    parser.add_argument(
        "--exp", choices=["A", "B", "C", "D", "S", "all"], default="all",
        help="Esperimento da eseguire (default: tutti)"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Genera solo i grafici dai CSV esistenti"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Sovrascrivi CSV esistenti senza chiedere conferma"
    )

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Simulatore LN — Tesi Ventilii 2025                ║")
    print("╚══════════════════════════════════════════════════════╝")

    if args.plot_only:
        print("  Modalità: solo grafici")
        generate_all_plots()
        from .runners.runner_scaling import plot_scaling
        plot_scaling()
        return

    exps = ["A", "B", "C", "D", "S"] if args.exp == "all" else [args.exp]
    print(f"  Esperimenti: {', '.join(exps)}")
    print(f"  Force: {args.force}")

    t0 = time.time()

    from .runners.runner_scaling import run_exp_scaling
    dispatch = {"A": run_exp_A, "B": run_exp_B,
                "C": run_exp_C, "D": run_exp_D,
                "S": run_exp_scaling}

    for e in exps:
        dispatch[e](force=args.force)

    elapsed = time.time() - t0
    print(f"\n{'═' * 54}")
    print(f"  Esperimenti completati in {elapsed:.0f}s")
    print(f"{'═' * 54}")

    # Genera grafici
    generate_all_plots()

    # Genera grafici scaling
    from .runners.runner_scaling import plot_scaling
    plot_scaling()


if __name__ == "__main__":
    main()
