import torch
import networkx as nx
import numpy as np
from state import *
from torch import randint


def topological_map_abstraction(embedding_func, graph, pool, trajectory):

    Candidate_queue = set([])
    TwoD_array_index = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    prob = np.ones(len(trajectory)**2)/len(trajectory)**2
    np.random.choice(TwoD_array_index, N, replace=True, p=prob)
    for i in range(N):
        pool.append(TwoD_array_index[i])
    opimizer.zero_grad()
    d.append(torch.norm(actual_embedding(
        pool[[i][0]])-actual_embedding(pool[[i][1]])))
    d_pred.append(torch.norm(pool[[i][0]])-embedding_func(pool[[i][1]]))
    loss = torch.sum(L(d, d_pred))
    loss.backward()
    optimizer.step()

    ArgminV_Index = torch.empty(len(trajectory))

    rho = torch.tensor(len(trajectory), len(Vertices))
    for i in range(len(trajectory)):
        for j in range(len(Vertices)):
            rho[i][j] = torch.norm(LocSEFunc(state)-LocSEFunc(l[j]))
        ArgminV_Index[i] = torch.argmin(rho[i])

        min_V = rho[i][ArgminV_Index[i]]
        if min_V < 1.5:
            trajectory[i].vertex = ArgminV_Index
        elif min_V > 2 and min_V < 3:
            Candidate_queue.append(trajectory[i])
        else:
            trajectory[i].vertex = NULL

    for S in Candidate_queue:
        graph.add_node(S)
        S.vertex = S
    for i in range(len(trajectory)):
        if trajectory[i].vertex != Null and trajectory[i+1].vertex != Null and trajectory[i].vertex != trajectory[i+1].vertex:
            graph.add_edge(trajectory[i], trajectory[i+1], weight=torch.norm(
                embedding_func(trajectory[i]))-embedding_func(trajectory[i+1]))
    for v1 in graph.vertices():
        for v2 in graph.vertices():
            if v2 == v1:
                pass
            elif torch.norm(embedding_func(v1)-embedding_func(v2)) < 1.5:
                graph.remove_node(np.random([v1, v2], 1, p=[0.5, 0.5]))
    for x in graph.edges():
        if graph.edges[x]['weight'] > 3:
            graph.remove_edge(x)
    # 9: end for
