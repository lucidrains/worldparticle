import torch
from torch.nn import Module, ModuleList

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class ParticleTransformer(Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
