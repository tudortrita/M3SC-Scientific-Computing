""" Simple bfs code from lecture 7 (corrected version)

    EDIT: 06/03/2019, Added deque type for Q
"""
import networkx as nx


def bfs(G=nx.barabasi_albert_graph(100, 5), s=0):
    """
    Input:
    G: networkx graph
    s: source node

    Output:
    L2: Labels for all nodes in graph, 0=unreachable from source, 1=reachable
    L3: Shortest distance from source to nodes in graph
    """
    from collections import deque

    L1 = list(G.nodes)  # Assumes nodes are numbered from 0 to N-1
    L2 = [0 for l in L1]
    L3 = [-1000 for l in L1]

    Q = deque()
    Q.append(s)
    L2[s] = 1
    L3[s] = 0

    while Q:
        x = Q.popleft()
        for v in G.adj[x].keys():
            if L2[v] == 0:
                Q.append(v)
                L2[v] = 1
                L3[v] = 1 + L3[x]
            # print("v=",v)
            # print("Q=",Q)
    return L2, L3


bfs()
