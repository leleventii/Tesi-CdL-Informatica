# Simulatore Lightning Network ⚡️

**Progetto di Tesi di Laurea in Informatica**  
*Autore: Emanuele Ventilii (2025)*

Questo repository contiene il codice sorgente per la simulazione e l'analisi topologica della **Lightning Network** (LN). Il progetto indaga le dinamiche di svuotamento dei canali (il cosiddetto effetto *"Leaking Bucket"*) e l'impatto dei nodi *Merchant* (ricevitori puri) sulla longevità della rete. Viene inoltre valutata l'efficacia di strategie di *Rebalancing* (riequilibrio della liquidità) per mitigare questi colli di bottiglia strutturali.

---

## 📂 Struttura del Progetto

Il codice è stato modularizzato e organizzato in un'architettura logica chiara per separare il motore di calcolo, l'esecuzione degli esperimenti e la generazione dei grafici.

```text
Main/
├── ln_sim/                     # Sorgenti Python del simulatore
│   ├── core.py                 # Motore di routing e logica di rebalancing
│   ├── graphs.py               # Generazione topologie (Clique, Dumbbell, ecc.)
│   ├── main.py                 # Entry point CLI (Interfaccia principale)
│   ├── runners/                # Script di esecuzione degli esperimenti
│   │   ├── runner_scaling.py   # Exp Scaling sulle Clique (tau vs n)
│   │   ├── runner.py           # Exp sul grafo Dumbbell e multi-parametro
│   │   ├── runner_snapshot.py  # Exp basati sulla topologia reale LN
│   │   └── runner_merchant_comparison.py # Confronto rebalancing su nodi reali
│   └── visualization/          # Script per il tracciamento dei grafici
│       ├── plots.py            # Generazione automatica grafici 2D (A, B, C, D)
│       ├── plot_3d.py          # Superficie 3D: tau vs (n, alpha)
│       ├── plot_3d_nk.py       # Superficie 3D: tau vs (n, k)
│       └── plot_wealth_transfer.py # Analisi dinamica del trasferimento di fondi
├── data/                       # Dataset e risultati grezzi in formato CSV
│   ├── dumbbell/               # CSV esperimenti grafi Dumbbell e multi-param
│   ├── scaling/                # CSV per lo scaling (tau in funzione di n)
│   └── snapshot/               # Dati sulle topologie reali ed estrazioni subgraph
├── results/                    # Output visivi della simulazione
│   ├── img/                    # Immagini finali e superfici 3D 
│   ├── plots_dumbbell/         # Plot generati dagli exp sul Dumbbell
│   ├── plots_scaling/          # Plot generati dagli exp sulle Clique
│   └── plots_snapshot/         # Plot derivati dagli esperimenti su topologia reale
└── archive_v2/                 # Script e dati di esperimenti obsoleti/scartati
```

---

## 🚀 Come avviare la simulazione

L'intero progetto può essere governato facilmente attraverso l'entry point centralizzato `ln_sim/main.py`.

### 1. Prerequisiti
Assicurati di aver attivato l'ambiente virtuale con le dipendenze installate (come `networkx`, `numpy`, `matplotlib`, `tqdm`).
```bash
source venv/bin/activate
# oppure per l'installazione delle dipendenze (se mancanti):
# pip install -r requirements.txt
```

### 2. Uso Base
Puoi lanciare gli esperimenti o la sola rigenerazione dei grafici specificando i parametri da terminale, avviando il modulo dalla *root* del progetto:

```bash
# Esegue tutti gli esperimenti e genera tutti i grafici
python -m ln_sim.main

# Visualizza i comandi e i flag di aiuto
python -m ln_sim.main --help
```

### 3. Flag Disponibili
- `--exp {A,B,C,D,S,all}` : Lancia un esperimento specifico.
  - `A`: Baseline (Variando il bias verso i merchant)
  - `B`: Scaling della capacità `k`
  - `C`: Studio dello spostamento dei colli di bottiglia (Dumbbell scatter)
  - `D`: Analisi della topologia LN Reale
  - `S`: Scaling puro del fallimento in funzione dei nodi `n`
- `--plot-only` : Non esegue alcuna simulazione, limitandosi a ricostruire i grafici dai file `.csv` precedentemente salvati in `data/`.
- `--force` : Forza la sovrascrittura dei vecchi file `csv` ignorando gli avvertimenti di sicurezza.

---

## 📊 Plotting 3D e Analisi Aggiuntive
Gli script avanzati per la visualizzazione 3D o per indagini microscopiche possono essere invocati autonomamente come moduli:

```bash
# Per generare le superfici 3D
python -m ln_sim.visualization.plot_3d
python -m ln_sim.visualization.plot_3d_nk

# Per visualizzare il flusso di fondi (Wealth Transfer)
python -m ln_sim.visualization.plot_wealth_transfer
```

---

## 🧪 Metodologia di Rebalancing
Il motore (`core.py`) implementa due modalità principali:
1. **Routing Base (Shortest Path)**: Le transazioni seguono percorsi ottimi non privilegiati, permettendo di misurare l'effetto fisiologico del "Leaking Bucket".
2. **Merchant-Driven Rebalancing**: Un algoritmo proattivo mirato a disinnescare la saturazione dei canali adiacenti ai nodi "Sink" (i commercianti), spostando la liquidità mediante pagamenti circolari. Tale logica dimostra limiti strutturali qualora la topologia non offra cicli di supporto dotati di liquidità sufficiente.
