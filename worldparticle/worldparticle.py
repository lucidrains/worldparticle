from __future__ import annotations
from collections import namedtuple

import torch
from torch import cat
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList

import einx
from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from torch_einops_utils import (
    pad_right_ndim_to_and_expand_as,
    pad_right_at_dim,
    pad_sequence,
    maybe,
    lens_to_mask,
)

# constants

TokenMergeOutput = namedtuple('TokenMergeOutput', ['tokens', 'pos', 'weights', 'lens'])

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

def gather_vectors(src, indices, dim = 1):
    expanded_indices = pad_right_ndim_to_and_expand_as(indices, src)
    return src.gather(dim, expanded_indices)

# proposed token merging algorithm in section 3.2

def merge_tokens(
    tokens,         # (b n d)
    pos = None,     # (b n 3)
    weights = None, # (b n)
    lens = None,    # (b)
):
    batch, seq_len, device = *tokens.shape[:2], tokens.device

    if not exists(lens):
        lens = torch.full((batch,), seq_len, device = device)

    # handle odd sequence length

    if is_odd(seq_len):
        tokens = pad_right_at_dim(tokens, 1, dim = 1) # make even

        weights = maybe(pad_right_at_dim)(weights, 1)
        pos = maybe(pad_right_at_dim)(pos, 1, dim = 1)

    even_seq_len = tokens.shape[1]

    # mask

    mask = lens_to_mask(lens, even_seq_len)

    weights = default(weights, 1.) * mask.float()

    # do the split they propose, which is just every other index

    src_tokens, tgt_tokens = rearrange(tokens, 'b (n two) d -> two b n d', two = 2)
    src_weights, tgt_weights = rearrange(weights, 'b (n two) -> two b n', two = 2)
    _, tgt_mask = rearrange(mask, 'b (n two) -> two b n', two = 2)

    if exists(pos):
        src_pos, tgt_pos = rearrange(pos, 'b (n two) p -> two b n p', two = 2)

    # they do cosine sim as the distance measure for merging

    with torch.no_grad():

        sim = einsum(l2norm(src_tokens), l2norm(tgt_tokens), 'b i d, b j d -> b i j')

        half_seq_len = even_seq_len // 2
        eye = torch.eye(half_seq_len, device = device, dtype = torch.bool)

        mask_value = -torch.finfo(sim.dtype).max

        sim = sim.masked_fill(eye, mask_value)
        sim = einx.where('b j, b i j,', tgt_mask, sim, mask_value)

        closest_match_index = sim.argmax(dim = -1) # (b i)

    # they merge the tokens, but keep track of the weights / mass, or the number of tokens merged within the super token
    # updates are weighted accordingly

    weighted_src_tokens = einx.multiply('b n d, b n', src_tokens, src_weights)
    weighted_tgt_tokens = einx.multiply('b n d, b n', tgt_tokens, tgt_weights)

    if exists(pos):
        weighted_src_pos = einx.multiply('b n p, b n', src_pos, src_weights)
        weighted_tgt_pos = einx.multiply('b n p, b n', tgt_pos, tgt_weights)

    closest_tgt_tokens = gather_vectors(weighted_tgt_tokens, closest_match_index)
    closest_tgt_weights = gather_vectors(tgt_weights, closest_match_index)

    merged_weighted_tokens = weighted_src_tokens + closest_tgt_tokens
    merged_weights = src_weights + closest_tgt_weights
    merged_tokens = einx.divide('b n d, b n', merged_weighted_tokens, merged_weights.clamp(min = 1e-5))

    if exists(pos):
        closest_tgt_pos = gather_vectors(weighted_tgt_pos, closest_match_index)
        merged_weighted_pos = weighted_src_pos + closest_tgt_pos
        merged_pos = einx.divide('b n p, b n', merged_weighted_pos, merged_weights.clamp(min = 1e-5))

    # handle the unmerged tokens

    tgt_mask_unmerged = tgt_mask.scatter(1, closest_match_index, False)

    unmerged_lens = tgt_mask_unmerged.sum(dim = -1).long()
    unmerged_lens_list = unmerged_lens.tolist()

    unmerged_tgt_tokens = tgt_tokens[tgt_mask_unmerged].split(unmerged_lens_list)
    unmerged_tgt_weights = tgt_weights[tgt_mask_unmerged].split(unmerged_lens_list)

    unmerged_tgt_tokens = pad_sequence(unmerged_tgt_tokens, dim = 0)
    unmerged_tgt_weights = pad_sequence(unmerged_tgt_weights, dim = 0)

    if exists(pos):
        unmerged_tgt_pos = tgt_pos[tgt_mask_unmerged].split(unmerged_lens_list)
        unmerged_tgt_pos = pad_sequence(unmerged_tgt_pos, dim = 0)

    # output are the merged tokens and unmerged target tokens

    output_tokens = cat((merged_tokens, unmerged_tgt_tokens), dim = 1)
    output_weights = cat((merged_weights, unmerged_tgt_weights), dim = 1)

    output_pos = cat((merged_pos, unmerged_tgt_pos), dim = 1) if exists(pos) else None

    output_lens = unmerged_lens + half_seq_len

    # return new tokens, weights, and the lengths

    return TokenMergeOutput(output_tokens, output_pos, output_weights, output_lens)

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
