import torch
import numpy as np
import State
from torch import randint
from playing_game import add_trajectory
from abstaction_map_to_V import *

def topological_map_abstraction(Pool, Trajectory, Vertices, Edges):
    # Algorithm 1
    # 1: Pool P←∅. Vertex set V←∅. Edge set E←∅. Graph G(V, E).
    Landmarks = set([])
    tradjectory_index = 0
    #Prob_distrb_s = []
    Candidate_queue = set([])
    # 4: Sample state pairs from T using distribution Ps and put them to P.
    # 5: Training the embedding function φθ using samples from P.
    for t_local_training in range(0, training_sessions):

    # 7: Add vertices and edges using
    ArgminV_Index = torch.empty(trajectory.shape[0])
    # [For each state s on a trajectory, we label it with its corresponding graph vertex.Let i= arg minj:vj ∈ V  ρ(φθ(s),φθ(lj)) and d = ρ(φθ(s),φθ(li)).  There are three possiblecases:  (1)d∈[0,1.5].  We label s with vi.  (2) d ∈ [2,3].  We consider s as an appropriate landmark candidate.  Therefore, we label s with NULL but add it to a candidate queue.  (3)Otherwise,sis simply labelled with NULL.]

    rho = torch.tensor(trajectory.shape[0],len(Vertices))
    for i in range(trajectory.shape[0]):
        for j in range(len(Vertices)):
            rho[i][j] = torch.norm(LocSEFunc(state)-LocSEFunc(l[j]))
        ArgminV_Index[i] = torch.argmin(rho[i])

        min_V = rho[i][ArgminV_Index[i]]
        if min_V < 1.5:
            state.vertex = v[i]

        else if min_V > 2 and min_V < 3:
            Candidate_queue.append(state)
            state.candidate_yes_no = True
            state.vertex = NULL
            state.candidate_label = ArgminV_index[i]
        else:
            state.vertex = NULL

    # [We move some states from the candidate queue into the landmark set and update V accordingly. Once a state is added to the landmark set, we will relabel it from NULL to its vertex identifier
    for state in Candidate_queue:
        state.vertex = state.candidate_label
    # [Let the labelled trajectory to be(vi0,vi1,...,vin). If we find vik and vik + 1 are different vertices in the existing graph, we will add an edge〈vik,vik+1〉into the graph]
    for i in range(shape(trajectory)[0]):
        
        # 8: Check the graph using
    for i in range(len(Vertices)):
        for j in range
    # [Ifρ(φθ(li),φθ(lj))<1.5, then we will merge vi and vj .]
    # [For any edge〈vi,vj〉, ifρ(φθ(li),φθ(lj))>3, we will remove this edge.]
    # 9: end for
