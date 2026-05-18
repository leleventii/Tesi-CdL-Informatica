import random
import numpy as np
import networkx as nx
import heapq
# ─────────────────────────────────────────────────
#  Pre-calcolo shortest path (una volta per grafo)
# ─────────────────────────────────────────────────

def precompute_paths(G):
    #Dato un grafo G, precalcola tutti gli shortest paths per ogni coppia di nodo
    #Restituisce un dizionario di paths
    nodi = list(G.nodes())
    paths = {}
    for sorgente in nodi:
        for destinazione in nodi:
            if sorgente != destinazione:
                paths[(sorgente, destinazione)] = list(nx.all_shortest_paths(G, sorgente, destinazione))
    return paths



def attempt_merchant_rebalance(failed, bal, eu, ev, M_set, G):
    """
    Tenta di eseguire il Rebalancing guidato dal Merchant.
    Quando un canale verso un merchant si svuota, il merchant orchestra un ciclo
    drenando liquidità da un cliente con bilancio favorevole e re-iniettandola
    nel canale svuotato.
    """
    resolved_all = True
    failed_edge = None

    for j in failed:
        u, v = int(eu[j]), int(ev[j])
        if min(bal[u, v], bal[v, u]) > 0:
            continue
            
        if bal[u, v] == 0:
            source, target = u, v
        else:
            source, target = v, u
            
        # source ha bilancio 0 verso target.
        if target in M_set:
            m = target
            c = source
        elif source in M_set:
            m = source
            c = target
        else:
            resolved_all = False
            failed_edge = (u, v) if u < v else (v, u)
            break
            
        # Il merchant 'm' cerca il vicino 'w' che ha più fondi verso di lui (cioè bal[m, w] minimo)
        neighbors = list(G.neighbors(m))
        if c in neighbors:
            neighbors.remove(c)
            
        if not neighbors:
            resolved_all = False
            failed_edge = (u, v) if u < v else (v, u)
            break
            
        w_opt = min(neighbors, key=lambda w: bal[m, w])
        
        # Percorso da c a w_opt escludendo gli altri merchant
        nodes_to_remove = M_set.difference({c, w_opt})
        G_sub = G.copy()
        G_sub.remove_nodes_from(nodes_to_remove)
        
        try:
            path_c_w = nx.shortest_path(G_sub, c, w_opt)
        except nx.NetworkXNoPath:
            resolved_all = False
            failed_edge = (u, v) if u < v else (v, u)
            break
            
        # Verifica liquidità reale per 1 Satoshi
        can_route = True
        if bal[m, c] < 1: can_route = False
        for i in range(len(path_c_w) - 1):
            x, y = path_c_w[i], path_c_w[i+1]
            if bal[x, y] < 1:
                can_route = False
                break
        if bal[w_opt, m] < 1: can_route = False
        
        if can_route:
            # Rebalance di 1 Satoshi
            bal[m, c] -= 1
            bal[c, m] += 1
            for i in range(len(path_c_w) - 1):
                x, y = path_c_w[i], path_c_w[i+1]
                bal[x, y] -= 1
                bal[y, x] += 1
            bal[w_opt, m] -= 1
            bal[m, w_opt] += 1
        else:
            resolved_all = False
            failed_edge = (u, v) if u < v else (v, u)
            break
            
    return resolved_all, failed_edge


# ─────────────────────────────────────────────────
#  Simulazione singola (Algorithm 1) — NumPy
# ─────────────────────────────────────────────────

def simulate(G, k, alpha, M, paths, seed=42, rebalance_mode="none"):
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
    rebalance_mode : str  'none', o 'merchant' per selezionare la strategia.

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
    M_set = set(m_list)

    # ── Inizializzazione Bilanci (Matrice NumPy n×n) ──
    # bal[u, v] rappresenta quanti bitcoin 'u' possiede e può ancora inviare verso 'v'.
    # Matrice int64 per accesso O(1) e check vettorizzato.
    bal = np.zeros((n, n), dtype=np.int64)

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
            if rebalance_mode == "none":
                j = failed[0]
                return tau, (int(eu[j]), int(ev[j]))
            elif rebalance_mode == "merchant":
                success, failed_edge = attempt_merchant_rebalance(failed, bal, eu, ev, M_set, G)
                if not success:
                    return tau, failed_edge

        # La transazione è stata incamerata senza svuotare nessun canale!
        tau += 1
