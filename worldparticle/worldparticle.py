import torch
from torch import cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, einsum

from torch_einops_utils import (
    pad_right_ndim_to_and_expand_as,
    pad_right_at_dim,
    pad_sequence,
    lens_to_mask,
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

    # mask

    mask = lens_to_mask(lens, even_seq_len)

    if not exists(weights):
        weights = mask.float()

    # do the split they propose, which is just every other index

    src_tokens, tgt_tokens = rearrange(tokens, 'b (n two) d -> two b n d', two = 2)
    src_weights, tgt_weights = rearrange(weights, 'b (n two) -> two b n', two = 2)
    src_mask, tgt_mask = rearrange(mask, 'b (n two) -> two b n', two = 2)

    # they do cosine sim

    sim = einsum(l2norm(src_tokens), l2norm(tgt_tokens), 'b i d, b j d -> b i j')

    half_seq_len = even_seq_len // 2
    eye = torch.eye(half_seq_len, device = device, dtype = torch.bool)

    mask_value = -torch.finfo(sim.dtype).max

    sim = sim.masked_fill(eye, mask_value)
    sim = einx.where('b j, b i j,', tgt_mask, sim, mask_value)

    closest_match_index = sim.argmax(dim = -1) # (b i)

    # they merge the tokens, but keep track of the weights / mass, or the number of tokens merged within the super token
    # updates are weighted accordingly

    expanded_closest_match_index = pad_right_ndim_to_and_expand_as(closest_match_index, tgt_tokens)

    weighted_src_tokens = einx.multiply('b n d, b n', src_tokens, src_weights)
    weighted_tgt_tokens = einx.multiply('b n d, b n', tgt_tokens, tgt_weights)

    closest_tgt_tokens = weighted_tgt_tokens.gather(1, expanded_closest_match_index)
    closest_tgt_weights = tgt_weights.gather(1, closest_match_index)

    merged_weighted_tokens = weighted_src_tokens + closest_tgt_tokens
    merged_weights = src_weights + closest_tgt_weights

    merged_tokens = einx.divide('b n d, b n', merged_weighted_tokens, merged_weights)

    # handle the unmerged tokens

    neg_ones = torch.ones_like(closest_match_index) * -1.
    tgt_mask_unmerged = tgt_mask.float().scatter_add(1, closest_match_index, neg_ones) > 0.

    unmerged_lens = tgt_mask_unmerged.sum(dim = -1).long()

    unmerged_tgt_tokens = tgt_tokens[tgt_mask_unmerged].split(unmerged_lens.tolist())
    unmerged_tgt_weights = tgt_weights[tgt_mask_unmerged].split(unmerged_lens.tolist())

    unmerged_tgt_tokens = pad_sequence(unmerged_tgt_tokens, dim = 0)
    unmerged_tgt_weights = pad_sequence(unmerged_tgt_weights, dim = 0)

    # output are the merged tokens and unmerged target tokens

    output_tokens = cat((merged_tokens, unmerged_tgt_tokens), dim = 1)
    output_weights = cat((merged_weights, unmerged_tgt_weights), dim = 1)

    output_lens = unmerged_lens + half_seq_len

    # return new tokens, weights, and the lengths

    return output_tokens, output_weights, output_lens

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
