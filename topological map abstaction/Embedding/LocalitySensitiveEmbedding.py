import numpy as np
import torch.nn.functional as F

# Constants
RC0 = 1
RC1 = 3
LAMBDA1 = 0.5
LAMBDA2 = 0.2

to_float16 = lambda x:np.dtype(np.float16).type(x)

def X1(x):
    return to_float16(0 <= x and x <= RC0)

def X2(x):
    return to_float16(RC0 < x and x <= RC1)

def X3(x):
    return to_float16(RC1 < x)

# Embedding Loss
def L1(x):
    return F.relu(x-RC0)

def L2(x):
    return F.relu(-x+RC0) + F.relu(x-RC1)

def L3(x):
    return F.relu(-x+RC1)

def L(d, d_expect):
    return X1(d)*L1(d_expect) + LAMBDA1*X2(d)*L2(d_expect) + LAMBDA2*X3(d)*L3(d_expect)

def embedding_func()