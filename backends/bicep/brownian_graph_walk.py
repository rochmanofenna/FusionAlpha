import networkx as nx
import numpy as _np
from .brownian_motion import simulate_single_path
from .stochastic_control import apply_stochastic_controls

def brownian_graph_walk(n_nodes, n_steps,
                        initial_state="00",
                        T=1.0,
                        directional_bias=0.0,
                        variance_adjustment=None):
    """
    Builds a simple 2-bit state‐graph and walks it using hybrid
    transitions driven by a Brownian path.
    """
    # 1) build 4-state graph
    G = nx.DiGraph()
    for s in ["00","01","10","11"]:
        G.add_node(s)
    edges = {
      ("00","01"):1.0,
      ("00","10"):1.0,
      ("01","11"):0.5,
      ("10","11"):0.5,
      ("11","00"):0.2
    }
    for (u,v),w in edges.items():
        G.add_edge(u,v,weight=w)

    # 2) generate a single Brownian path
    dt = T/n_steps
    path = simulate_single_path(
        T, n_steps, 0, dt,
        directional_bias, variance_adjustment,
        _np, apply_stochastic_controls
    )

    # 3) walk
    cur = initial_state
    history = [cur]
    import random
    for i in range(n_steps):
        delta = path[i+1] - path[i]
        nbrs = list(G.neighbors(cur))
        if not nbrs:
            break
        # weighted choice
        ws = [G[cur][v]["weight"]*(1+delta) for v in nbrs]
        tot = sum(ws)
        probs = [w/tot for w in ws]
        cur = random.choices(nbrs, probs)[0]
        history.append(cur)

    # 4) update weights
    counts = {s:history.count(s) for s in G.nodes()}
    for u in G.nodes():
        tot = sum(counts[v] for v in G.neighbors(u))
        for v in G.neighbors(u):
            if tot>0:
                G[u][v]["weight"] = counts[v]/tot

    return G, history
