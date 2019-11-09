"""Dijkstra algorithm implemented using dictionaries.
Note: Implementation is different from lecture 8 slides but similar
to the in-class example.
"""
import networkx as nx


def dijkstra(G, s=0):
    """Find shortest distances to s in weighted graph, G"""

    # Initialize dictionaries
    dinit = 10**6
    Edict = {}  # Explored nodes
    Udict = {}  # Uneplroed nodes

    for n in G.nodes():
        Udict[n] = dinit
    Udict[s] = 0

    # Main search
    while Udict:
        # Find node with min d in Udict and move to Edict
        dmin = dinit
        for n, w in Udict.items():
            if w < dmin:
                dmin = w
                nmin = n
        Edict[nmin] = Udict.pop(nmin)
        print("moved node", nmin)

        # Update provisional distances for unexplored neighbors of nmin
        for n, w in G.adj[nmin].items():
            if n in Udict:
                dcomp = dmin + w['weight']
                if dcomp < Udict[n]:
                    Udict[n] = dcomp

    return Edict


if __name__ == '__main__':
    # Example from lecture 8 slides
    e = [[1, 2, 3], [1, 5, 2], [2, 3, 1], [
        2, 5, 1], [5, 4, 5], [3, 4, 2], [4, 6, 1]]
    G_in = nx.Graph()
    G_in.add_weighted_edges_from(e)
    Edict_out = dijkstra(G_in, 1)
