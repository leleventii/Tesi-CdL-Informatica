# Resoconto Progetto Tesi: Simulatore "Lightning Network" (Ventilii 2025)

Questo documento sintetizza l'intero stato del progetto della tesi. È stato creato per poter essere fornito come contesto iniziale (prompt) a qualsiasi istanza di Gemini per fargli capire immediatamente l'architettura, gli obiettivi e a che punto è il lavoro.

## 1. Obiettivo della Tesi
Analizzare empiricamente la stabilità e la vulnerabilità topologica della Lightning Network (LN) e, più in generale, dei Payment Channel Networks (PCN). Lo studio si concentra in particolare su:
- **L'effetto "Secchio Bucato" (Leaking Bucket / Sink-Effect)**: Dimostrare come l'immissione di ulteriore liquidità (Aumento della capacità $k$) non prevenga il fallimento della rete quando vi è un orientamento direzionale dei pagamenti (Merchant Bias $\alpha$). La legge di scala del tempo di fallimento ($\tau$) passa da $O(k^2)$ nel caso uniforme a $O(k)$ con la presenza di merchant.
- **Topologie**: Dalle cricche pure (Clique), ai grafi a manubrio (Dumbbell per i colli di bottiglia), fino a reti complesse (Barabási-Albert, Erdős-Rényi) e l'analisi sul **vero Snapshot della Lightning Network Reale**.
- **JIT Circular Rebalancing**: Analisi dell'impatto algoritmico del rebalancing circolare Just-In-Time e dei suoi effetti "parassitari" sulla liquidità globale.
- **Transizioni di Fase**: Calcolo della probabilità critica di bias $\alpha^*$ oltre la quale la rete collassa deterministicamente a causa del collo di bottiglia strutturale.

## 2. Architettura del Codice
Il progetto è diviso in script modulari scritti in **Python** (con largo uso di `NumPy`, `NetworkX` e `Multiprocessing` per scalare sul silicio hardware M4 di Apple). La cartella principale è `/ln_sim/`.

### Core Engine
* `ln_sim/core.py`: Il cuore matematico. Implementa "Algorithm 1" del paper di riferimento. **Super ottimizzato con NumPy** per evitare i loop di Python; utilizza matrici $n \times n$ (int64) per bilanci e calcoli vettorializzati vettori. Supporta anche la ricerca "Widest Path" per il rebalancing (Intelligent JIT Rebalancing).
* `ln_sim/graphs.py`: Generatori topologici. Costruisce le topologie sintetiche (Clique, Dumbbell, Barabasi-Albert, Erdos-Renyi) ed espone routine matematiche per layout di rete.

### Motori di Simulazione (Runners)
I runner utilizzano `multiprocessing.Pool` con *initializer* globale per eludere i blocchi di pickling (serializzazione degli oggetti grossi in memoria RAM) e permettere l'utilizzo ottimizzato multi-threading:
* `ln_sim/runner.py`: Esegue gli esperimenti base.
  * **EXP A** ($\tau$ vs $\alpha$): Declino vita della rete al variare del bias $\alpha$.
  * **EXP B** ($\tau$ vs $k$): Analisi log-log delle leggi di scala spaziali (Il celebre grafico "Secchio bucato").
  * **EXP C** (Dumbbell scatter): Studio microscopico dello spostamento del bilancio.
  * **EXP D** (Transizione $\alpha^*$): Scoperta e calcolo del punto critico per phase transition.
* `ln_sim/runner_scaling.py` ed `ln_sim/runner_complex.py`: Studiano l'andamento asintotico di $\tau$ all'aumentare dei nodi ($n$) simulando i comportamenti centralizzati vs decentralizzati.
* `ln_sim/runner_snapshot.py`: L'esperimento sulla "Realtà". Ingerisce snapshot reali di nodi LN (JSON pubblici), estrae un sottografo matematico che ingloba i Top 60-100 hub e fa girare le routine di routing, provando la fragilità logica della vera Lightning Network basata sul parametro "Merchant Bias".

### Visualizzazione (Plots)
Il layout produce risultati sia in Raw Data (CSV su diverse folder `data-*/`) per studio successivo, sia in Output visuali diretti pronti per il pre-print accademico.
* I vari runner integrano script `matplotlib` integrati.
* Esitono pipeline come `plot_3d.py`/`plot_3d_nk.py` per proiezioni rendering 3D sulle superfici multidimensionali.

## 3. Stato di Avanzamento Attuale (Aprile 2026)
1. **Setup Ottimizzato (NumPy/M4)**: Completamente refactorato nei round precedenti. Algoritmi ottimizzati matematicamente rimuovendo bottleneck con i loop in Python puro.
2. **"Grafico B" (Bucket Effect) Formalizzato**: Risolti a monte i problemi sui calcoli dei cicli ed esplosione teorica delle pendenze ($k^2 \to k$), documentato empiricamente.
3. **Studio Snapshot Isolato e Funzionante**: Il branch di testing LN Reale `runner_snapshot.py` scansiona attivamente i JSON, esegue pipeline isolate senza spaccare il setup sintetico, gestendo gli hub e generando i suoi plot per tesi comparativi.
4. **Grafica per Tesi**: Creati intervalli di confidenza di Wilson e customizzazioni dei diagrammi logaritmici molto accurate.

## 4. Prompt Consigliato da usare nelle Nuove Chat
Copia e incolla in blocco questo messaggio per una sessione ex-novo pulita:

> Ciao Gemini! Sto tornando a lavorare alla mia Tesi del 2025 sulla simulazione matematica della fallibilità della Lightning Network ("Secchio Bucato"). Come setup base ho gli script Python in `ln_sim/` orchestrati da diversi `runner*.py` suddivisi tra simulazioni base, complessi, di scalabilità e di Snapshot della rete reale. Il core computazionale usa `NetworkX` per i routing path ed una matrice NumPy ultra-veloce (`core.py`) per monitorare la liquidità dei nodi lungo i check con Algorithm 1 (incluso rebalancing circolare algoritmico basato sui Widest Path). Attualmente tutto il backend è validato e completato con risultati csv robusti messi su `/data`. 
> 
> La mia richiesta / focus per la fase di oggi è: [INSERISCI COSA DEVO FARE]
