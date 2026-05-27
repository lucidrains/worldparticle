from __future__ import annotations
from collections import namedtuple

import torch
from torch import cat, Tensor, tensor
from torch.nn import Linear, Module, ModuleList, RMSNorm, Sequential
import torch.nn.functional as F

import einx
from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from x_mlps_pytorch import create_mlp

from torch_einops_utils import (
    pad_right_ndim_to_and_expand_as,
    pad_right_at_dim,
    pad_sequence,
    maybe,
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

def gather_vectors(src, indices, dim = 1):
    expanded_indices = pad_right_ndim_to_and_expand_as(indices, src)
    return src.gather(dim, expanded_indices)

# 3d axial rotary embeddings

class AxialRotaryEmbeddings(Module):
    def __init__(
        self,
        dim,
        omega = 10_000
    ):
        super().__init__()
        assert divisible_by(dim, 6), f'{dim} must be divisible by 6'
        inv_freq = omega ** (-torch.arange(0, dim, 6).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    def forward(
        self,
        pos, # (... 3)
    ):
        freqs = einsum(pos, self.inv_freq, '... p, f -> ... p f')
        freqs = rearrange(freqs, 'b ... p f -> b 1 ... (p f)')
        return cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# proposed token merging algorithm in section 3.2

TokenMergeOutput = namedtuple('TokenMergeOutput', ['tokens', 'pos', 'weights', 'lens'])

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
        tokens = pad_right_at_dim(tokens, 1, dim = 1)

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

    # handle unmerged tokens

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

    half_orig_lens = (lens + 1) // 2
    output_lens = unmerged_lens + half_orig_lens

    # return new tokens, weights, and the lengths

    return TokenMergeOutput(output_tokens, output_pos, output_weights, output_lens)

# swiglu - Shazeer et al.

class SwiGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.silu(gate)

# feedforward

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4
    ):
        super().__init__()
        dim_inner = int(dim * mult * 2 / 3)
        self.net = Sequential(
            Linear(dim, dim_inner * 2),
            SwiGLU(),
            Linear(dim_inner, dim)
        )

    def forward(self, x):
        return self.net(x)

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
        context = None,
        context_mask = None,
        rotary_emb = None,
        context_rotary_emb = None
    ):
        context = default(context, tokens)

        queries, keys, values = (
            self.to_queries(tokens),
            *self.to_keys_values(context).chunk(2, dim = -1)
        )

        queries, keys, values = (self.split_heads(t) for t in (queries, keys, values))

        if exists(rotary_emb):
            queries = apply_rotary_pos_emb(rotary_emb, queries)
            keys = apply_rotary_pos_emb(default(context_rotary_emb, rotary_emb), keys)

        queries = queries * self.scale

        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        if exists(context_mask):
            sim = einx.where('b j, b h i j,', context_mask, sim, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, values, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        return self.to_out(out)

# film

class FiLM(Module):
    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = RMSNorm(dim, elementwise_affine = False)

        self.to_gamma_beta = Linear(dim_cond, dim * 2, bias = False)
        torch.nn.init.zeros_(self.to_gamma_beta.weight)

    def forward(
        self,
        tokens,
        cond
    ):
        normed = self.norm(tokens)

        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)

        scaled = einx.multiply('b n d, b n d', normed, gamma + 1.)
        return einx.add('b n d, b n d', scaled, beta)

# classes

CorrectorOutput = namedtuple('CorrectorOutput', ['pos', 'vel'])

class ParticleTransformerCorrector(Module):
    def __init__(
        self,
        dim,
        enc_depth,
        dec_depth,
        enc_dim_head = 64,
        enc_heads = 8,
        dec_dim_head = 64,
        dec_heads = 8,
        ff_mult = 4,
        pred_dim_hidden = 512,
        pred_num_layers = 5,
        film_context_with_weights = False,
        film_cond_dim = None,
    ):
        super().__init__()

        assert divisible_by(enc_dim_head, 6), f'enc_dim_head ({enc_dim_head}) must be divisible by 6 for 3d axial rotary embeddings'
        self.axial_rotary_emb = AxialRotaryEmbeddings(enc_dim_head)

        # determine encoder-decoder layer matching
        # match as many as possible, pad with last if decoder is deeper
        # take the last dec_depth encoder layers if encoder is deeper

        if dec_depth <= enc_depth:
            enc_layer_indices = list(range(enc_depth - dec_depth, enc_depth))
        else:
            enc_layer_indices = list(range(enc_depth)) + [enc_depth - 1] * (dec_depth - enc_depth)

        self.register_buffer('enc_layer_indices', tensor(enc_layer_indices))

        # super token encoder - self attention -> token merge -> feed forward

        self.enc_layers = ModuleList([])

        for _ in range(enc_depth):
            self.enc_layers.append(ModuleList([
                RMSNorm(dim),
                Attention(dim, dim_head = enc_dim_head, heads = enc_heads),
                RMSNorm(dim),
                FeedForward(dim, mult = ff_mult),
            ]))

        # super token decoder - cross attention -> self attention -> feed forward

        self.dec_layers = ModuleList([])

        for _ in range(dec_depth):
            self.dec_layers.append(ModuleList([
                RMSNorm(dim),
                RMSNorm(dim),
                Attention(dim, dim_head = dec_dim_head, heads = dec_heads),
                RMSNorm(dim),
                Attention(dim, dim_head = dec_dim_head, heads = dec_heads),
                RMSNorm(dim),
                FeedForward(dim, mult = ff_mult),
            ]))

        # optional film conditioning of cross attention context tokens with super token weights

        self.film_context_with_weights = film_context_with_weights

        if film_context_with_weights:
            film_cond_dim = default(film_cond_dim, dim)

            self.weight_to_cond = Sequential(
                Rearrange('... -> ... 1'),
                Linear(1, film_cond_dim * 2),
                SwiGLU(),
                Linear(film_cond_dim, film_cond_dim)
            )

            self.films = ModuleList([FiLM(dim, film_cond_dim) for _ in range(dec_depth)])

        # final norm and prediction head

        self.final_norm = RMSNorm(dim)

        self.to_pred = create_mlp(
            pred_dim_hidden,
            depth = pred_num_layers - 1,
            dim_in = dim,
            dim_out = 6
        )

    def forward(
        self,
        tokens,
        pos,
        weights = None,
        lens = None
    ):
        # super token encoder

        enc_tokens, enc_pos, enc_weights, enc_lens = tokens, pos, weights, lens

        enc_intermediates = []

        for (
            attn_norm,
            attn,
            ff_norm,
            ff
        ) in self.enc_layers:
            rotary_emb = self.axial_rotary_emb(enc_pos)

            # self attention

            enc_tokens = attn(attn_norm(enc_tokens), rotary_emb = rotary_emb) + enc_tokens

            # proposed token merging using cosine sim

            enc_tokens, enc_pos, enc_weights, enc_lens = merge_tokens(
                enc_tokens,
                pos = enc_pos,
                weights = enc_weights,
                lens = enc_lens
            )

            # feedforward

            enc_tokens = ff(ff_norm(enc_tokens)) + enc_tokens

            enc_intermediates.append((enc_tokens, enc_pos, enc_weights, enc_lens))

        # super token decoder

        dec_tokens = tokens
        rotary_emb = self.axial_rotary_emb(pos)

        for ind, (dec_layer, enc_layer_index) in enumerate(zip(self.dec_layers, self.enc_layer_indices)):

            (
                cross_context_norm,
                cross_norm,
                cross_attn,
                self_norm,
                self_attn,
                ff_norm,
                ff
            ) = dec_layer

            # the decoder attends to successively merged super particle tokens

            layer_enc_tokens, layer_enc_pos, layer_enc_weights, layer_enc_lens = enc_intermediates[enc_layer_index]

            context_rotary_emb = self.axial_rotary_emb(layer_enc_pos)
            context_mask = lens_to_mask(layer_enc_lens, layer_enc_tokens.shape[1])

            if self.film_context_with_weights:
                context_cond = self.weight_to_cond(layer_enc_weights)
                context = self.films[ind](layer_enc_tokens, context_cond)
            else:
                context = cross_context_norm(layer_enc_tokens)

            dec_tokens = cross_attn(
                cross_norm(dec_tokens),
                context = context,
                context_mask = context_mask,
                rotary_emb = rotary_emb,
                context_rotary_emb = context_rotary_emb
            ) + dec_tokens

            # self attention

            dec_tokens = self_attn(
                self_norm(dec_tokens),
                rotary_emb = rotary_emb
            ) + dec_tokens

            # feedforward

            dec_tokens = ff(ff_norm(dec_tokens)) + dec_tokens

        # predict position and velocity residuals

        pos_vel = self.to_pred(self.final_norm(dec_tokens))
        pos_residual, vel_residual = pos_vel.chunk(2, dim = -1)

        return CorrectorOutput(pos_residual, vel_residual)
