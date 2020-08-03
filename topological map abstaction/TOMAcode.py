import torch
import networkx as nx
import numpy as np
import State
from torch import randint
from playing_game import add_trajectory


def topological_map_abstraction(embedding_func, graph, pool, trajectory):
    # Algorithm 1

    #Prob_distrb_s = []
    Candidate_queue = set([])
    # 4: Sample state pairs from T using distribution Ps and put them to P.
    TwoD_array_index = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    prob = np.ones(len(trajectory), len(trajectory))/len(trajectory)**2
    np.random.choice(TwoD_array_index, N, replace=True, p=prob)
    for i in range(N):
        pool.append(TwoD_array_index[i])
    # 5: Training the embedding function φθ using samples from P.
    opimizer.zero_grad()
    d.append(torch.norm(actual_embedding(
        pool[[i][0]])-actual_embedding(pool[[i][1]])))
    d_pred.append(torch.norm(pool[[i][0]])-embedding_func(pool[[i][1]]))
    loss = torch.sum(L(d, d_pred))
    loss.backward()
    optimizer.step()

    # 7: Add vertices and edges using
    ArgminV_Index = torch.empty(len(trajectory))
    # [For each state s on a trajectory, we label it with its corresponding graph vertex.Let i= arg minj:vj ∈ V  ρ(φθ(s),φθ(lj)) and d = ρ(φθ(s),φθ(li)).  There are three possiblecases:  (1)d∈[0,1.5].  We label s with vi.  (2) d ∈ [2,3].  We consider s as an appropriate landmark candidate.  Therefore, we label s with NULL but add it to a candidate queue.  (3)Otherwise,sis simply labelled with NULL.]

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

    # [We move some states from the candidate queue into the landmark set and update V accordingly. Once a state is added to the landmark set, we will relabel it from NULL to its vertex identifier
    for S in Candidate_queue:
        graph.add_node(S)
        S.vertex = S
    # [Let the labelled trajectory to be(vi0,vi1,...,vin). If we find vik and vik + 1 are different vertices in the existing graph, we will add an edge〈vik,vik+1〉into the graph]
    for i in range(len(trajectory)):
        if trajectory[i].vertex != Null and trajectory[i+1].vertex != Null and trajectory[i].vertex != trajectory[i+1].vertex:
            graph.add_edge(trajectory[i], trajectory[i+1])
            graph.edges[trajectory[i], trajectory[i+1]]['distance'] = torch.norm(
                embedding_func(trajectory[i]))-embedding_func(trajectory[i+1])
    # 8: Check the graph using
    for v1 in graph.vertices():
        for v2 in graph.vertices():
            if v2 == v1:
                pass
            elif torch.norm(embedding_func(v1)-embedding_func(v2)) < 1.5:
                graph.remove_node(np.random([v1, v2], 1, p=[0.5, 0.5]))
    for x in graph.edges():
        if graph.edges[x]['distance'] > 3:
            graph.remove_edge(x)
    # 9: end for
