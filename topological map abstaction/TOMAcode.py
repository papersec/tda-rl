import torch
import numpy as np
from torch import randint
from playing_game import add_trajectory
from abstaction_map_to_V import *

# Algorithm 1
# 1: Pool P←∅. Vertex set V←∅. Edge set E←∅. Graph G(V, E).
Pool = []
Vertices = set([])
Edges = set([])
Landmarks = set([])
t = 1
tradjectory_index = 0
Prob_distrb_s = []
training_sessions = 0
Candidate_queue = []
# 2: for t = 1, 2, ... do
for (t > 0):
# 3: Sample a trajectory T using some policy π or by random. or with some policy π

# 4: Sample state pairs from T using distribution Ps and put them to P.
# 5: Training the embedding function φθ using samples from P.
for t_local_training in range(0, training_sessions):

    # 6: Initialize G using
if len(Vertices) == 0:


# [If V=∅, we will pick a landmark from currently sampled trajectories and add a vertex into V accordingly. In our implementation, this landmark is the initial state of the agent.]
# if it’s empty.
# 7: Add vertices and edges using
ArgminV_Index = torch.empty(len(trajectory))
# [For each state s on a trajectory, we label it with its corresponding graph vertex.Let i= arg minj:vj ∈ V  ρ(φθ(s),φθ(lj)) and d = ρ(φθ(s),φθ(li)).  There are three possiblecases:  (1)d∈[0,1.5].  We label s with vi.  (2) d ∈ [2,3].  We consider s as an appropriate landmark candidate.  Therefore, we label s with NULL but add it to a candidate queue.  (3)Otherwise,sis simply labelled with NULL.]

rho = torch.tensor(len(trajectory),len(Vertices))
for i in range(len(trajectory)):
    for j in range(len(Vertices)):
        rho[i][j] = torch.norm(LocSEFunc(state)-LocSEFunc(l[j]))
    ArgminV_Index[i] = torch.argmin(rho[i])

    min_V = rho[i][ArgminV_Index[i]]
    if abc < 1.5:
        pass

    else if abc > 2 and abc < 3:
        Candidate_queue.append()
    else:


    # [We move some states from the candidate queue into the landmark set andupdateVaccordingly. Once a state is added to the landmark set, we will relabel it from NULL toits vertex identifier
# [Let the labelled trajectory to be(vi0,vi1,...,vin). If we find vik and vik + 1 are different vertices in the existing graph, we will add an edge〈vik,vik+1〉into the graph]

    # 8: Check the graph using
for i in range(len(Vertices)):
    for j in range
# [Ifρ(φθ(li),φθ(lj))<1.5, then we will merge vi and vj .]
# [For any edge〈vi,vj〉, ifρ(φθ(li),φθ(lj))>3, we will remove this edge.]
# 9: end for


