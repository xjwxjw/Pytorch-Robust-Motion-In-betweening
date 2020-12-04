import torch
import numpy as np
import torch.nn as nn

def PLU(x, alpha = 0.1, c = 1.0):
    relu = nn.ReLU()
    o1 = alpha * (x + c) - c
    o2 = alpha * (x - c) + c
    o3 = x - relu(x - o2)
    o4 = relu(o1 - o3) + o3
    return o4

def gen_ztta(dim = 256, length = 50):
    ### currently without T_max ###
    ztta = np.zeros((1, length, dim))
    for t in range(length):
        for d in range(dim):
            if d % 2 == 0:
                ztta[:, t, d] = np.sin(1.0 * (length - t) / 10000 ** (d / dim))
            else:
                ztta[:, t, d-1] = np.cos(1.0 * (length - t) / 10000 ** (d / dim))
    return torch.from_numpy(ztta.astype(np.float))

def gen_ztar(sigma = 1.0, length = 50):
    ### currently noise term in not inroduced ###
    lambda_tar = []
    for t in range(length):
        if t < 5:
            lambda_tar.append(0)
        elif t < 30 and t >= 5:
            lambda_tar.append((t - 5.0) / 25.0)
        else:
            lambda_tar.append(1)
    lambda_tar = np.array(lambda_tar)
    return torch.from_numpy(lambda_tar)