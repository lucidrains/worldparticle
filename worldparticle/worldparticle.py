from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn
from torch import cat, is_tensor, stack, Tensor, tensor
from torch.nn import Linear, Module, ModuleList, RMSNorm, Sequential, Parameter
import torch.nn.functional as F

import einx
from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from x_mlps_pytorch import create_mlp

from torch_einops_utils import (
    pad_right_ndim_to,
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

def gather_neighbors(
    src,        # (b n d)
    indices     # (b m k)
):              # -> (b m k d)
    b = src.shape[0]
    batch_seq = torch.arange(b, device = src.device)
    batch_indices = pad_right_ndim_to(batch_seq, indices.ndim)
    return src[batch_indices, indices]

def derive_neighbors_from_radius(
    pos_y,                  # (b m d)
    pos_x,                  # (b n d)
    r,
    max_num_neighbors
):                          # -> (b m k), (b m k)
    try:
        from torch_cluster import radius
    except ImportError:
        raise ImportError('torch-cluster must be installed to derive neighbors dynamically. Install with: pip install torch-cluster')

    b, m, device = *pos_y.shape[:2], pos_y.device
    _, n = pos_x.shape[:2]

    pos_y_flat = rearrange(pos_y, 'b m d -> (b m) d')
    pos_x_flat = rearrange(pos_x, 'b n d -> (b n) d')

    batch_y = torch.arange(b, device = device).repeat_interleave(m)
    batch_x = torch.arange(b, device = device).repeat_interleave(n)

    edge_index = radius(
        pos_x_flat,
        pos_y_flat,
        r = r,
        batch_x = batch_x,
        batch_y = batch_y,
        max_num_neighbors = max_num_neighbors
    )

    dst_indices, src_indices = edge_index[0], edge_index[1]

    dst_indices, perm = dst_indices.sort()
    src_indices = src_indices[perm]

    src_indices = src_indices % n

    counts = torch.bincount(dst_indices, minlength = b * m)
    max_k = min(int(counts.max().item()), max_num_neighbors)

    if max_k == 0:
        return pos_y.new_zeros(b, m, 0, dtype = torch.long), pos_y.new_zeros(b, m, 0, dtype = torch.bool)

    mask = lens_to_mask(counts, max_len = max_k)

    out_indices = pos_y.new_zeros(b * m, max_k, dtype = torch.long)
    out_indices[mask] = src_indices

    out_indices = rearrange(out_indices, '(b m) k -> b m k', b = b)
    out_mask = rearrange(mask, '(b m) k -> b m k', b = b)

    return out_indices, out_mask

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

        self.to_queries_gates = Linear(dim, dim_inner * 2, bias = False)
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

        queries, gates, keys, values = (
            *self.to_queries_gates(tokens).chunk(2, dim = -1),
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

        out = out * gates.sigmoid()
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

        # the decoder attends to the final merged super particle tokens

        layer_enc_tokens, layer_enc_pos, layer_enc_weights, layer_enc_lens = enc_intermediates[-1]

        context_rotary_emb = self.axial_rotary_emb(layer_enc_pos)
        context_mask = lens_to_mask(layer_enc_lens, layer_enc_tokens.shape[1])

        for ind, dec_layer in enumerate(self.dec_layers):

            (
                cross_context_norm,
                cross_norm,
                cross_attn,
                self_norm,
                self_attn,
                ff_norm,
                ff
            ) = dec_layer

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

# non-neural predictor

PredictorOutput = namedtuple('PredictorOutput', ['pos', 'vel'])

class ParticlePredictor(Module):
    def __init__(
        self,
        delta_time = 1.
    ):
        super().__init__()
        self.register_buffer('delta_time', tensor(delta_time))

    def forward(
        self,
        pos,
        vel,
        mass = None,
        forces = None
    ):
        dt = self.delta_time

        # section 3, equation 1: explicit prediction step

        vel_pred = vel

        if exists(forces):
            assert exists(mass)
            accel = einx.divide('b n p, b n', forces, mass.clamp(min = 1e-8))
            vel_pred = vel_pred + dt * accel

        pos_pred = pos + (dt / 2.) * (vel + vel_pred)

        return PredictorOutput(pos_pred, vel_pred)

# 3d lattice learnable kernel

class LearnableKernel3D(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        grid_res = 5,
        radius = 1.
    ):
        super().__init__()
        self.radius = radius
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.theta = Parameter(torch.randn(1, dim_in * dim_out, grid_res, grid_res, grid_res) * (dim_in ** -0.5))

    def forward(
        self,
        rel_pos,    # (b n k 3)
        features    # (b n k di)
    ):              # -> (b n k do)

        b, n, k = rel_pos.shape[:3]

        grid = rearrange(rel_pos / self.radius, 'b n k c -> 1 1 1 (b n k) c')

        weights = F.grid_sample(self.theta, grid, mode = 'bilinear', padding_mode = 'zeros', align_corners = True)
        weights = rearrange(weights, '1 (i o) 1 1 p -> p i o', i = self.dim_in)

        features = rearrange(features, 'b n k i -> (b n k) i')
        out = einsum(features, weights, 'p i, p i o -> p o')

        return rearrange(out, '(b n k) o -> b n k o', b = b, n = n, k = k)

# particle tokenizer

class ParticleTokenizer(Module):
    def __init__(
        self,
        dim,
        dim_attr = 0,
        dim_boundary_attr = 0,
        grid_res = 5,
        spatial_radius = 1.,
        boundary_radius = 1.,
        topo_radius = 1.,
        max_spatial_neighbors = 32,
        max_boundary_neighbors = 32,
        mlp_depth = 2,
        dim_hidden = None,
    ):
        super().__init__()
        self.max_spatial_neighbors = max_spatial_neighbors
        self.max_boundary_neighbors = max_boundary_neighbors

        dim_hidden = default(dim_hidden, dim)

        self.kernel_spatial  = LearnableKernel3D(3 + dim_attr, dim, grid_res, spatial_radius)
        self.kernel_boundary = LearnableKernel3D(max(dim_boundary_attr, 1), dim, grid_res, boundary_radius)
        self.kernel_topo     = LearnableKernel3D(3 + dim_attr + 3, dim, grid_res, topo_radius)

        dim_mlp_in = dim * 3 + 3 + dim_attr
        self.to_tokens = create_mlp(dim_hidden, depth = mlp_depth, dim_in = dim_mlp_in, dim_out = dim)

    def forward(
        self,
        pos,                        # (b, n, 3) - predicted positions x̃
        vel,                        # (b, n, 3) - predicted velocities ṽ
        attrs = None,               # (b, n, ca) - per-particle attributes c
        pos_rest = None,            # (b, n, 3) - rest-pose positions x^0
        spatial_indices = None,     # (b, n, ks) - spatial neighbor indices
        spatial_mask = None,        # (b, n, ks)
        boundary_pos = None,        # (b, m, 3) - boundary positions x^b
        boundary_attrs = None,      # (b, m, cb) - boundary attributes c^b
        boundary_indices = None,    # (b, n, kb) - boundary neighbor indices
        boundary_mask = None,       # (b, n, kb)
        topo_indices = None,        # (b, n, kt) - topological neighbor indices
        topo_mask = None,           # (b, n, kt)
    ):
        b, n, device = *pos.shape[:2], pos.device

        if exists(attrs):
            attrs = pad_right_ndim_to(attrs, 3)

        if exists(boundary_attrs):
            boundary_attrs = pad_right_ndim_to(boundary_attrs, 3)

        attrs = default(attrs, pos.new_empty(b, n, 0))

        def aggregate(kernel, rel_pos, features, mask = None):
            out = kernel(rel_pos, features)
            if exists(mask):
                out = einx.multiply('b n k d, b n k', out, mask)
            return out.sum(dim = 2)

        pos_i = rearrange(pos, 'b n d -> b n 1 d')

        # spatial branch - particle-particle interactions

        if not exists(spatial_indices) and exists(pos):
            spatial_indices, spatial_mask = derive_neighbors_from_radius(pos, pos, self.kernel_spatial.radius, self.max_spatial_neighbors)

        if exists(spatial_indices):
            pos_j   = gather_neighbors(pos, spatial_indices)
            vel_j   = gather_neighbors(vel, spatial_indices)
            attrs_j = gather_neighbors(attrs, spatial_indices)

            spatial_agg = aggregate(self.kernel_spatial, pos_j - pos_i, cat((vel_j, attrs_j), dim = -1), spatial_mask)
        else:
            spatial_agg = pos.new_zeros(b, n, self.kernel_spatial.dim_out)

        # boundary branch - particle-boundary interactions

        if not exists(boundary_indices) and exists(boundary_pos):
            boundary_indices, boundary_mask = derive_neighbors_from_radius(pos, boundary_pos, self.kernel_boundary.radius, self.max_boundary_neighbors)

        if exists(boundary_indices) and exists(boundary_pos):
            pos_j_b = gather_neighbors(boundary_pos, boundary_indices)

            attrs_j_b = gather_neighbors(boundary_attrs, boundary_indices) if exists(boundary_attrs) else pos.new_ones(*boundary_indices.shape, 1)

            boundary_agg = aggregate(self.kernel_boundary, pos_j_b - pos_i, attrs_j_b, boundary_mask)
        else:
            boundary_agg = pos.new_zeros(b, n, self.kernel_boundary.dim_out)

        # topology branch - rest-shape guided interactions

        if exists(topo_indices) and exists(pos_rest):
            pos_j      = gather_neighbors(pos, topo_indices)
            vel_j      = gather_neighbors(vel, topo_indices)
            attrs_j    = gather_neighbors(attrs, topo_indices)
            pos_rest_j = gather_neighbors(pos_rest, topo_indices)

            rel_pos_rest = einx.subtract('b n k d, b n d', pos_rest_j, pos_rest)
            rel_pos_curr = einx.subtract('b n k d, b n d', pos_j, pos)

            topo_agg = aggregate(self.kernel_topo, rel_pos_rest, cat((vel_j, attrs_j, rel_pos_curr), dim = -1), topo_mask)
        else:
            topo_agg = pos.new_zeros(b, n, self.kernel_topo.dim_out)

        return self.to_tokens(cat((spatial_agg, boundary_agg, topo_agg, vel, attrs), dim = -1))

# main module

WorldParticleOutput = namedtuple('WorldParticleOutput', ['pos', 'vel'])

class WorldParticle(Module):
    def __init__(
        self,
        *,
        predictor: ParticlePredictor | dict | None = None,
        corrector: ParticleTransformerCorrector | dict,
        tokenizer: Module | None = None
    ):
        super().__init__()
        self.tokenizer = tokenizer

        predictor = default(predictor, dict())

        self.predictor = ParticlePredictor(**predictor) if isinstance(predictor, dict) else predictor
        self.corrector = ParticleTransformerCorrector(**corrector) if isinstance(corrector, dict) else corrector

    def forward(
        self,
        *,
        pos,                            # (b n 3)
        vel,                            # (b n 3)
        tokens = None,                  # (b n d) | (b steps n d)
        mass = None,                    # (b n)
        forces = None,                  # (b n 3) | (b steps n 3)
        weights = None,                 # (b n)
        lens = None,                    # (b)
        num_steps = None,               # ()
        return_initial_state = False,
        tokenizer_kwargs: dict = dict()
    ):
        return_trajectory = exists(num_steps) or return_initial_state
        has_tokenizer = exists(self.tokenizer)

        # local functions

        def is_tensor_with_time(t):
            return is_tensor(t) and t.ndim == 4

        def to_iterable(t, has_time):
            if has_time:
                return t.unbind(dim = 1)
            if isinstance(t, (list, tuple)):
                return t
            return (t,) * num_steps

        forces_has_time = is_tensor_with_time(forces)
        tokens_has_time = not has_tokenizer and is_tensor_with_time(tokens)

        # auto-infer num steps

        if not exists(num_steps):
            num_steps = forces.shape[1] if forces_has_time else (tokens.shape[1] if tokens_has_time else 1)
            return_trajectory = (num_steps > 1) or return_initial_state

        # unpack time dimension if exists, else repeat

        forces = to_iterable(forces, forces_has_time)
        tokens = to_iterable(tokens, tokens_has_time)

        # rollout

        curr_pos, curr_vel = pos, vel
        positions, velocities = [], []

        for step_forces, step_tokens in zip(forces, tokens):

            pred_pos, pred_vel = self.predictor(
                pos = curr_pos,
                vel = curr_vel,
                mass = mass,
                forces = step_forces
            )

            if has_tokenizer:
                step_tokens = self.tokenizer(pos = pred_pos, vel = pred_vel, **tokenizer_kwargs)

            assert exists(step_tokens), 'tokens must be provided if tokenizer is not available'

            pos_residual, vel_residual = self.corrector(
                tokens = step_tokens,
                pos = pred_pos,
                weights = weights,
                lens = lens
            )

            curr_pos = pred_pos + pos_residual
            curr_vel = pred_vel + vel_residual

            positions.append(curr_pos)
            velocities.append(curr_vel)

        if return_initial_state:
            positions.insert(0, pos)
            velocities.insert(0, vel)

        # stack

        positions, velocities = (stack(t, dim = 1) for t in (positions, velocities))

        if not return_trajectory:
            positions, velocities = positions[:, 0], velocities[:, 0]

        return WorldParticleOutput(positions, velocities)
