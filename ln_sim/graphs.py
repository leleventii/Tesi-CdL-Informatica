"""
graphs.py — Costruzione grafi e analisi di rete
================================================
Fornisce funzioni per generare le strutture topologiche specifiche
usate negli esperimenti empirici del progetto.
Fornisce anche metriche critiche per l'analisi del degrado (es. betweenness).
"""

import networkx as nx


# ─────────────────────────────────────────────────
#  Generatori di topologia (Strutture di Rete)
# ─────────────────────────────────────────────────

def make_clique(n):
    """
    Genera un grafo completo (Clique) di dimensione 'n' (K_n).
    - Proprietà: Massimo parallelismo e decentralizzazione possibile.
    - Path length: Qualsiasi nodo è connesso a qualsiasi altro nodo con 1 singolo hop.
    - Utilizzo base: Test teorico "best case scenario" (espansione massima).
    """
    return nx.complete_graph(n)


def make_ring(n):
    """
    Genera un grafo a ciclo chiuso (Ring) di dimensione 'n' (C_n).
    - Proprietà: Struttura a bottleneck persistente.
    - Path length: O(n), ogni transazione attraversa multipli colli di bottiglia.
    """
    return nx.cycle_graph(n)


def make_star(n):
    """
    Genera un grafo a stella di 'n' nodi totali.
    - Hub: Nodo centrale identificato dall'indice 0 (Degree = n-1)
    - Foglie: Nodi da 1 a n-1 (Degree = 1)
    - Utilizzo base: Modellizzazione di provider fortemente centralizzati.
    """
    return nx.star_graph(n - 1)


def make_dumbbell(c=5):
    """
    Genera il grafo "Dumbbell" (manubrio): costituito da due Clique bilanciate
    K_c che sono tenute assieme da ESATTAMENTE un singolo arco denominato "Ponte".

    - Lato sinistro  (Clique) : Nodi [0, 1, ..., c-1]
    - Lato destro (Clique)    : Nodi [c, c+1, ..., 2c-1]
    - Ponte (Arco di rottura) : Arco che collega il nodo c-1 al nodo c.
    
    Questo grafo è specializzato per l'Esperimento sulla "Transizione di Fase":
    oppone un collo di bottiglia macroscopico-topologico (il Ponte) a un collo 
    di bottiglia microscopico-stocastico (il nodo Merchant definito a indice 0).
    """
    G = nx.Graph()
    # Genera Clique sinistra
    for i in range(c):
        for j in range(i + 1, c):
            G.add_edge(i, j)
            
    # Genera Clique destra
    for i in range(c, 2 * c):
        for j in range(i + 1, 2 * c):
            G.add_edge(i, j)
            
    # Erigi l'Infrastruttura del Ponte (Collega le due isole topologiche)
    G.add_edge(c - 1, c)
    return G


def make_random_regular(n, d):
    """
    Genera un grafo random d-regolare di 'n' nodi.
    Ogni nodo ha esattamente grado 'd'.
    - Proprietà: Uniformità topologica perfetta — nessun nodo è privilegiato
      dalla struttura. Se emerge il sink-effect, è PURO da α.
    - Vincolo: n*d deve essere pari (altrimenti il grafo non esiste).
    - Utilizzo: Contrappunto alla Clique per dimostrare universalità del sink-effect.
    """
    if (n * d) % 2 != 0:
        raise ValueError(f"n*d deve essere pari: n={n}, d={d}, n*d={n*d}")
    if d >= n:
        raise ValueError(f"d deve essere < n: d={d}, n={n}")
    return nx.random_regular_graph(d, n)


def make_barabasi_albert(n, d):
    """
    Genera un grafo scale-free di Barabási-Albert di 'n' nodi.
    Per far sì che il grado medio sia circa 'd', ogni nuovo nodo 
    si attacca a m = d//2 nodi esistenti.
    
    Nota fondamentale per LN: la funzione di default di NetworkX fa sì che
    i primi nodi aggiunti (es. 0, 1, 2...) attirino naturalmente 
    la maggioranza delle connessioni diventando enormi Hub topologici.
    """
    m = max(1, d // 2)
    if m >= n:
        m = n - 1
    return nx.barabasi_albert_graph(n, m)


def make_erdos_renyi(n, d):
    """
    Genera un grafo puramente random di Erdős-Rényi di 'n' nodi.
    La probabilità p è calcolata per ottenere un grado medio atteso pari a 'd'.
    
    Siccome questo grafo potrebbe generare componenti disconnesse (che 
    farebbero crashare la ricerca degli shortest path), estraiamo
    direttamente il Giant Component connesso più grande, scartando i rami morti.
    """
    p = min(1.0, d / (n - 1))
    
    # Per essere certi che sia connesso, iteriamo finchè non otteniamo un single component
    # o prendiamo il G.C. In LN sim preferiamo che mantenga i nodi intatti.
    # Faremo try-retry veloce con seed diversi finchè non genera un grafo connesso.
    max_retries = 100
    for _ in range(max_retries):
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G
            
    # Se fallisce dopo 100 tentativi (raro per d elevati, ma possibile per d bassi)
    # prendiamo la connected component maggiore
    G = nx.erdos_renyi_graph(n, p)
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc).copy()
    
    # Rinomina i nodi da 0 a n_sub-1 per mantenere la compatibilità della struttura
    return nx.convert_node_labels_to_integers(G_sub)


# ─────────────────────────────────────────────────
#  Analisi ed Estrazione Metriche di Rete
# ─────────────────────────────────────────────────

def get_edge_betweenness(G):
    """
    Calcola la centralità di Betweenness per TUTTI gli archi della rete.
    Questa metrica dice statisticamente quante volte uno shortest path 
    attraversa ciascun arco. Usata per tracciare la fragilità topologica di un arco.

    Restituisce: Un dizionario {(u, v): betweenness_score} normalizzato in formato
                 canonico (u < v) indipendente dal senso dell'arco.
    """
    # Computazione raw su networkx (non scalato da n * n-1 per non perdere i numeri puri)
    raw = nx.edge_betweenness_centrality(G, normalized=False)
    result = {}
    for (u, v), val in raw.items():
        a, b = (u, v) if u < v else (v, u)
        result[(a, b)] = val
    return result


def classify_edge(u, v, M, bridge_edge):
    """
    Classifica topologicamente lo "scopo" di un arco per l'analisi nel grafo di test Dumbbell.
    Serve nei plot per categorizzare i dati.

    Parameters
    ----------
    u, v         : indici dei due nodi estremità dell'arco in questione.
    M            : insieme di nodi con ruolo 'Merchant'.
    bridge_edge  : tupla che descrive l'arco isolato di interconnessione macro.

    Returns:  'bridge' | 'merchant_adj' | 'other'
    """
    # Standardizza il verso dell'arco
    a, b = (u, v) if u < v else (v, u)
    
    # Ordine di priorità nell'analisi: Prima si accerta se è il fulcro geografico principale
    if (a, b) == bridge_edge:
        return "bridge"
        
    # Se almeno uno dei due nodi dell'arco tocca un Merchant (assorbe il sink di destinazione)
    if a in M or b in M:
        return "merchant_adj"
        
    # Archi neutri interni (es. archi dentro la cricca di destra)
    return "other"

