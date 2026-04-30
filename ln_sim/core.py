"""
core.py — Motore di simulazione PCN (Ottimizzato NumPy)
=========================================================
Implementa esattamente Algorithm 1 del paper arXiv:2511.16376.

Concetti chiave:
- τ (tau): numero di pagamenti completati prima che la liquidità di un qualsiasi arco 
  scenda a zero (bmin(e) == 0).
- Il pagamento viene SEMPRE eseguito forzatamente, anche se l'ultimo step porta il bilancio a 0.
- Il routing è topologico puro: ignora completamente lo stato dei bilanci, simulando
  così il comportamento "miope" del routing di Lightning Network.

OTTIMIZZAZIONE:
  I bilanci sono mantenuti in una matrice NumPy n×n (int64) anziché in un dizionario Python.
  Il check di failure usa np.minimum vettorizzato (C-level) anziché un for-loop Python
  con ~5000 dict lookups. Speedup stimato: 3-5× sul hot path per n≥50.
"""

import random
import numpy as np
import networkx as nx


# ─────────────────────────────────────────────────
#  Pre-calcolo shortest path (una volta per grafo)
# ─────────────────────────────────────────────────

def precompute_paths(G):
    """
    Calcola in anticipo tutti gli shortest path (percorsi minimi) per ogni
    coppia sorgente-destinazione (s, d) con s != d.

    Ottimizzazione: facciamo questo calcolo una sola volta all'avvio dell'esperimento
    invece che ad ogni singola transazione, risparmiando enormi quantità di tempo.

    Returns
    -------
    dict  paths[(s, d)] = lista di path, dove ogni path è semplicemente una lista ordinata di nodi.
    """
    nodes = list(G.nodes())
    paths = {}
    for s in nodes:
        for d in nodes:
            if s != d:
                # nx.all_shortest_paths restituisce tutti i percorsi es aequo più corti
                paths[(s, d)] = list(nx.all_shortest_paths(G, s, d))
    return paths


# ─────────────────────────────────────────────────
#  Algoritmo di Rebalancing Probabilistico (Widest Path su C/2)
# ─────────────────────────────────────────────────

def get_best_cycle(C_half, source, target, n):
    """
    Trova il percorso con la massima capacità stimata (Widest Path probabilistico) da source a target.
    Usa la matrice statica C_half (C[u,v]/2) per garantire la privacy dei bilanci reali.
    """
    dist = np.zeros(n, dtype=np.int64)
    dist[source] = 999999999
    parent = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    
    for _ in range(n):
        # np.where vettorizzato per ignorare i nodi già visitati
        unvisited_dist = np.where(visited, -1, dist)
        u = np.argmax(unvisited_dist)
        max_d = unvisited_dist[u]
                
        if max_d <= 0 or u == target:
            break
            
        visited[u] = True
        
        cap = C_half[u, :]
        proposed = np.minimum(dist[u], cap)
        
        update_mask = (~visited) & (proposed > dist)
        dist[update_mask] = proposed[update_mask]
        parent[update_mask] = u
        
    if dist[target] == 0:
        return None, 0
        
    path = []
    curr = target
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    return path, dist[target]


def attempt_rebalance(failed, bal, C_half, eu, ev, n, k):
    """
    Tenta di eseguire il JIT Circular Rebalancing Probabilistico per i canali falliti.
    Restituisce (True, None) se il rebalancing ha successo per tutti i canali,
    oppure (False, failed_edge) se fallisce.
    """
    resolved_all = True
    failed_edge = None

    for j in failed:
        u, v = int(eu[j]), int(ev[j])
        if min(bal[u, v], bal[v, u]) > 0:
            continue  # Già risolto da un rebalance precedente
        
        # Identifichiamo la direzione svuotata: source -> target
        if bal[u, v] == 0:
            source, target = u, v
        else:
            source, target = v, u
            
        # BUG FIX CRITICO: Siccome C_half è statica, penserà sempre che l'arco diretto 
        # source -> target sia capiente, e ci restituirà sempre quello come "Miglior Percorso"!
        # Ma l'arco diretto è proprio quello svuotato! Dobbiamo "spegnerlo" temporaneamente
        # per costringere l'algoritmo a cercare un percorso ALTERNATIVO (il vero ciclo).
        temp_cap = C_half[source, target]
        C_half[source, target] = 0
            
        # Cerchiamo il path più capiente stimato (usando C_half statica, privacy-preserving)
        path_reb, est_cap = get_best_cycle(C_half, source, target, n)
        
        # Ripristiniamo la matrice per le prossime transazioni
        C_half[source, target] = temp_cap
        
        if path_reb is None or est_cap < 1:
            resolved_all = False
            failed_edge = (u, v) if u < v else (v, u)
            break
            
        # Proviamo a spostare l'importo stimato (limitato da k)
        amount = min(k, est_cap)
        
        # VERIFICA SUI BILANCI REALI
        # Controlliamo se i bilanci effettivi supportano la transazione.
        # Se non la supportano, il probe fallisce e la rete si ferma.
        can_route = True
        for i in range(len(path_reb) - 1):
            x, y = path_reb[i], path_reb[i + 1]
            if bal[x, y] < amount:
                can_route = False
                break
                
        if can_route:
            # Rebalance! Spostiamo liquidità lungo il ciclo
            # 1. Routing lungo il network
            for i in range(len(path_reb) - 1):
                x, y = path_reb[i], path_reb[i + 1]
                bal[x, y] -= amount
                bal[y, x] += amount
                
            # 2. Chiusura del ciclo sul canale diretto per ristabilire i fondi
            bal[target, source] -= amount
            bal[source, target] += amount
        else:
            # Il rebalancing fallisce perché i bilanci reali non supportano la stima
            resolved_all = False
            failed_edge = (u, v) if u < v else (v, u)
            break
            
    return resolved_all, failed_edge


# ─────────────────────────────────────────────────
#  Simulazione singola (Algorithm 1) — NumPy
# ─────────────────────────────────────────────────

def simulate(G, k, alpha, M, paths, seed=42, rebalance_active=False):
    """
    Esegue UNA singola simulazione della rete di canali di pagamento (PCN).

    Parameters
    ----------
    G      : nx.Graph   Grafo non orientato che rappresenta la topologia.
    k      : int        Liquidità iniziale su ciascun lato del canale (capacità totale 2k).
    alpha  : float      Probabilità bias: con probabilità alpha la destinazione è un Merchant.
    M      : list       Lista dei nodi classificati come Merchant.
    paths  : dict       Dizionario dei percorsi minimi pre-calcolati (da precompute_paths).
    seed   : int        Seme per il generatore di numeri pseudocasuali, garantisce la riproducibilità.
    rebalance_active : bool  Se True, abilita il JIT Circular Rebalancing probabilistico privacy-preserving.

    Returns
    -------
    tau : int
        Numero di pagamenti completati con successo PRIMA che il primo canale si esaurisca.
    first_failed_edge : tuple
        L'arco (u, v) formattato con u < v che ha causato l'arresto (ha raggiunto bmin == 0).
    """
    # Imposta il seme randomico per garantire la ripetibilità esatta di questo set di parametri
    random.seed(seed)

    nodes = list(G.nodes())
    n = len(nodes)
    m_list = list(M)

    # ── Inizializzazione Bilanci (Matrice NumPy n×n) ──
    # bal[u, v] rappresenta quanti bitcoin 'u' possiede e può ancora inviare verso 'v'.
    # Matrice int64 per accesso O(1) e check vettorizzato.
    bal = np.zeros((n, n), dtype=np.int64)
    
    # Matrice statica C_half per rebalancing probabilistico (C[u,v] / 2)
    C_half = np.zeros((n, n), dtype=np.int64)

    # Array di indici degli archi canonici (u < v) per il check vettorizzato
    eu_list = []  # indici u di ogni arco
    ev_list = []  # indici v di ogni arco

    for u, v in G.edges():
        a, b = (u, v) if u < v else (v, u)
        
        if 'capacity' in G[a][b]:
            cap_val = G[a][b]['capacity'] // 2
        else:
            cap_val = k
            
        bal[a, b] = k
        bal[b, a] = k
        C_half[a, b] = cap_val
        C_half[b, a] = cap_val
        
        eu_list.append(a)
        ev_list.append(b)

    # Converti in array numpy per fancy indexing vettorizzato
    eu = np.array(eu_list, dtype=np.intp)
    ev = np.array(ev_list, dtype=np.intp)

    tau = 0  # Contatore delle transazioni andate a buon fine

    # ── Loop principale della transazione (Algorithm 1) ──
    while True:
        # ── Step 1: Selezione della sorgente (Uniforme) ──
        # Chiunque nella rete può decidere di pagare, inclusi i merchant.
        s = random.choice(nodes)

        # ── Step 2: Selezione della destinazione (Con bias α) ──
        if random.random() < alpha and m_list:
            # Caso 2A: Transazione asimmetrica (Probabilità α) -> Destinazione = Merchant
            d = random.choice(m_list)
            
            # Un nodo non ripaga se stesso (s != d)
            while d == s and len(m_list) > 1:
                d = random.choice(m_list)
                
            # Fallback di sicurezza: se l'unico merchant della rete è chi sta pagando,
            # lo trattiamo eccezionalmente come una transazione normale.
            if d == s:
                d = s
                while d == s:
                    d = random.choice(nodes)
        else:
            # Caso 2B: Transazione uniforme (Probabilità 1-α) -> Destinazione = chiunque
            d = s
            while d == s:
                d = random.choice(nodes)

        # Scudo antierrore logico
        if s == d:
            continue

        # ── Step 3: Selezione del percorso P (Topologico) ──
        path = random.choice(paths[(s, d)])

        # ── Step 4: Esecuzione del pagamento lungo il path ──
        # Aggiornamento diretto sulla matrice numpy (accesso O(1) per cella)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            bal[u, v] -= 1
            bal[v, u] += 1

        # ── Step 5: Verifica condizione di guasto (Vettorizzata) ──
        # np.minimum su array → loop C-level anziché Python: ~5× più veloce per n≥50.
        bmin = np.minimum(bal[eu, ev], bal[ev, eu])
        failed = np.flatnonzero(bmin == 0)
        
        if failed.size > 0:
            if not rebalance_active:
                j = failed[0]
                return tau, (int(eu[j]), int(ev[j]))
            else:
                success, failed_edge = attempt_rebalance(failed, bal, C_half, eu, ev, n, k)
                if not success:
                    return tau, failed_edge

        # La transazione è stata incamerata senza svuotare nessun canale!
        tau += 1

