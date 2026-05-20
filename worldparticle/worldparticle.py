import torch
from torch.nn import Module, ModuleList

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_queries = Linear(dim, dim_inner, bias = False)
        self.to_keys_values = Linear(dim, dim_inner * 2, bias = False)

        self.to_out = Linear(dim_inner, dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        tokens,
        context = None
    ):
        context = default(context, tokens)

        queries, keys, values = (
            self.to_queries(tokens),
            *self.to_keys_values(context).chunk(2, dim = -1)
        )

        queries, keys, values = (self.split_heads(t) for t in (queries, keys, values))

        queries = queries * self.scale

        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, values, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        return self.to_out(out)

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
