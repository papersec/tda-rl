import torch
import numpy as np
from torch import randint
from playing_game import add_trajectory
from abstaction_map_to_V import *

# Algorithm 1
# 1: Pool P←∅. Vertex set V←∅. Edge set E←∅. Graph G(V, E).
Pool = []
Vertices = []
Edges = set([])
Landmarks = []
Trajectorys = []
t_global = 1
t_local_training = 0
tradjectory_index = 0
Prob_distrb_s = []
training_sessions = 0
Candidate_queue = []
# 2: for t = 1, 2, ... do
for (t_global > 0):
    # 3: Sample a trajectory T using some policy π or by random. or with some policy π
Trajectorys.append(add_trajectory())
# 4: Sample state pairs from T using distribution Ps and put them to P.
# 5: Training the embedding function φθ using samples from P.
for t_local_training in range(0, training_sessions):

    # 6: Initialize G using
if len(Vertices) == 0
R = abs(randint())
Vertices.append(Trajectorys[R])
# [If V=∅, we will pick a landmark from currently sampled trajectories and add a vertex into V accordingly. In our implementation, this landmark is the initial state of the agent.]
# if it’s empty.
# 7: Add vertices and edges using
StateIndex = torch.empty(1,len(Trajectorys[R]))
rho[i] = torch.norm(LocSEFunc(state)-LocSEFunc(v[i]))
for j in range(0, len(Vertices)-1):
    Rho[j+1] = torch.cat(Rho[j],rho[j+1])

# [For each state s on a trajectory, we label it with its corresponding graph vertex.Let i= arg minj:vj ∈ V  ρ(φθ(s),φθ(lj)) and d = ρ(φθ(s),φθ(li)).  There are three possiblecases:  (1)d∈[0,1.5].  We label s with vi.  (2) d ∈ [2,3].  We consider s as an appropriate landmark candidate.  Therefore, we label s with NULL but add it to a candidate queue.  (3)Otherwise,sis simply labelled with NULL.]
for i in range(0,len(Trajectorys[R])):
    StateIndex[i] = torch.argmin(Rho)
    if Rho[torch.argmin(Rho)] in range(0,1.5) :

    else if Rho[torch.argmin(Rho)] in range (2,3) :
        Candidate_queue.append()
    else :

# [We move some states from the candidate queue into the landmark set and update V accordingly. Once a state is added to the landmark set, we will relabel it from NULL to its vertex identifier.]

# [Let the labelled trajectory to be(vi0,vi1,...,vin). If we find vik and vik + 1 are different vertices in the existing graph, we will add an edge〈vik,vik+1〉into the graph]

    # 8: Check the graph using
for i in range(0, len(Vertices)):
    for j in range
# [Ifρ(φθ(li),φθ(lj))<1.5, then we will merge vi and vj .]
# [For any edge〈vi,vj〉, ifρ(φθ(li),φθ(lj))>3, we will remove this edge.]
# 9: end for


# Algorithm 2
# 1:fort= 1,2,...do
# 2:Set a goalgusing some criterion.
# 3:Sample a trajectory T under the guidance of intermediate goals.
# 4:Update graph using T (Algorithm 1).
# 5:Update vertex memory and HER using T.
# 6:Train the policy π using experience drawn from vertex memory and HER.
# 7:end for
