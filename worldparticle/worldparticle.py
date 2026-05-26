import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, einsum

from torch_einops_utils import (
    pad_right_ndim_to_and_expand_as,
    pad_right_at_dim,
    lens_to_mask
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

# proposed token merging algorithm in section 3.2

def merge_tokens(
    tokens,         # (b n d)
    weights = None, # (b n)
    lens = None,    # (b)
):
    batch, seq_len, device = *tokens.shape[:2], tokens.device

    if not exists(lens):
        lens = torch.full((batch,), seq_len, device = device)

    # handle odd sequence length

    if is_odd(seq_len):
        tokens = pad_right_at_dim(tokens, 1, dim = 1) # make even

        if exists(weights):
            weights = pad_right_at_dim(weights, 1)

    even_seq_len = tokens.shape[1]

    if not exists(weights):
        weights = torch.ones((batch, even_seq_len), device = device)

    # mask

    mask = lens_to_mask(lens, even_seq_len)

    # do the split they propose, which is just every other index

    src_tokens, tgt_tokens = rearrange(tokens, 'b (n two) d -> two b n d', two = 2)
    src_mask, tgt_mask = rearrange(mask, 'b (n two) -> two b n', two = 2)

    src_weights, tgt_weights = rearrange(weights, 'b (n two) -> two b n', two = 2)

    # they do cosine sim

    sim = einsum(l2norm(src_tokens), l2norm(tgt_tokens), 'b i d, b j d -> b i j')

    eye = torch.eye(even_seq_len // 2, device = device, dtype = torch.bool)

    sim = sim.masked_fill(eye, -torch.finfo(sim.dtype).max)

    closest_match_index = sim.argmax(dim = -1) # (b i)

    expanded_closest_match_index = pad_right_ndim_to_and_expand_as(closest_match_index, tgt_tokens)

    weighted_src_tokens = einx.multiply('b n d, b n', src_tokens, src_weights)
    weighted_tgt_tokens = einx.multiply('b n d, b n', tgt_tokens, tgt_weights)

    merged_weighted_tokens = weighted_src_tokens.scatter_add(1, expanded_closest_match_index, weighted_tgt_tokens) 
    denom = src_weights.scatter_add(1, closest_match_index, tgt_weights)

    merged_tokens = einx.divide('b n d, b n', merged_weighted_tokens, denom)

    return merged_tokens, weights, lens

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
