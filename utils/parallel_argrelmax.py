"""
This is a small function for implementing the torch version (parallel) of np.argrelmax function, requested by Omar
Ben Ren, 2019.04.16
It simply uses 2 convolution and relu with one multiply in the end to get the index of local minima
Note that this local minima has to be strictly larger than both sides (no flat area is allowed)
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def torch_argrelmax(input_tensor):
    up_kernel = torch.tensor([-1, 1, 0], dtype=torch.float).view(1, 1, -1)
    down_kernel = torch.tensor([0, 1, -1], dtype=torch.float).view(1, 1, -1)
    huge_padding = 9999999. * torch.ones([np.shape(input_tensor)[0], 1, 1])
    padded_tensor = torch.cat([huge_padding, input_tensor, huge_padding], dim=2)
    up_branch = F.conv1d(input=padded_tensor, weight=up_kernel, stride=1, bias=None, padding=0)
    down_branch = F.conv1d(input=padded_tensor, weight=down_kernel, stride=1, bias=None, padding=0)
    return 1 * (F.relu(up_branch) * F.relu(down_branch) != 0)


if __name__ == '__main__':
    a = torch.tensor([100,10,20,30,20,10,20,10,20,30,40,50,1,2], dtype=torch.float).view(2,1,-1)
    print(a)
    print(torch_argrelmax(a))