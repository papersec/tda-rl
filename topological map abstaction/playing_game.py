import numpy as np
live = bool
state = torch.empty(N,N)

def add_trajectory():
    trajectory = []
    step = (state, action)

    while live:
    # 열심히 게임을 한다.
    trajectory.append(state)
    return trajectory
