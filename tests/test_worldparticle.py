import pytest
param = pytest.mark.parametrize

import torch
from torch import nn

try:
    import torch_cluster
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False

from worldparticle.worldparticle import merge_tokens, ParticleTransformerCorrector, ParticleTokenizer

@param('has_pos', (False, True))
def test_merge_tokens(
    has_pos
):
    tokens = torch.randn(2, 63, 16)
    pos = torch.randn(2, 63, 3) if has_pos else None
    lens = torch.tensor((63, 31))
    weights = None

    for _ in range(10):
        tokens, pos, weights, lens = merge_tokens(tokens, pos, weights, lens)

    assert (lens == 1).all()
    assert (weights[:, 0] >= torch.tensor([63., 31.])).all()

    # already 1 token

    tokens = torch.ones(1, 1, 16)
    pos = torch.randn(1, 1, 3)
    tokens_out, pos_out, weights_out, lens_out = merge_tokens(tokens, pos)

    assert (lens_out == 1).all()
    assert (weights_out == 1.).all()
    assert torch.allclose(tokens_out, tokens)

    # batch with masked out targets

    tokens = torch.ones(2, 3, 16)
    pos = torch.randn(2, 3, 3)
    weights = torch.ones(2, 3)
    lens = torch.tensor((1, 3))

    tokens_out, pos_out, weights_out, lens_out = merge_tokens(tokens, pos, weights = weights, lens = lens)

    assert lens_out[0] == 1
    assert weights_out[0, 0] == 1.
    assert weights_out[0, 1] == 0.

@param('film_context_with_weights', (False, True))
def test_corrector(film_context_with_weights):
    corrector = ParticleTransformerCorrector(
        dim = 16,
        enc_depth = 2,
        dec_depth = 2,
        enc_dim_head = 6,
        enc_heads = 2,
        dec_dim_head = 6,
        dec_heads = 2,
        film_context_with_weights = film_context_with_weights,
    )

    tokens = torch.randn(2, 63, 16)
    pos = torch.randn(2, 63, 3)
    lens = torch.tensor((63, 31))

    pos_delta, vel_delta = corrector(tokens, pos = pos, lens = lens)

    assert pos_delta.shape == (2, 63, 3)
    assert vel_delta.shape == (2, 63, 3)

def test_predictor():
    from worldparticle.worldparticle import ParticlePredictor
    predictor = ParticlePredictor(delta_time = 0.01)

    tokens = torch.randn(2, 63, 16)
    pos = torch.randn(2, 63, 3)
    vel = torch.randn(2, 63, 3)
    forces = torch.randn(2, 63, 3)
    mass = torch.ones(2, 63)

    pos_pred, vel_pred = predictor(pos = pos, vel = vel, forces = forces, mass = mass)

    assert pos_pred.shape == (2, 63, 3)
    assert vel_pred.shape == (2, 63, 3)

def test_worldparticle_rollout():
    from worldparticle.worldparticle import WorldParticle, ParticlePredictor, ParticleTransformerCorrector, ParticleTokenizer
    from torch import nn

    corrector_kwargs = dict(
        dim = 16,
        enc_depth = 2,
        dec_depth = 2,
        enc_dim_head = 6,
        enc_heads = 2,
        dec_dim_head = 6,
        dec_heads = 2
    )

    predictor = ParticlePredictor(delta_time = 0.01)

    tokenizer = ParticleTokenizer(
        dim = 16,
        dim_attr = 1,
        grid_res = 5
    )

    model = WorldParticle(
        predictor = predictor,
        corrector = corrector_kwargs,
        tokenizer = tokenizer
    )

    pos = torch.randn(2, 63, 3)
    vel = torch.randn(2, 63, 3)
    forces = torch.randn(2, 63, 3)
    mass = torch.ones(2, 63)
    lens = torch.tensor((63, 31))

    # default single step - no time dim in output

    out = model(pos = pos, vel = vel, mass = mass, forces = forces, lens = lens, attrs = mass)

    assert out.pos.shape == (2, 63, 3)
    assert out.vel.shape == (2, 63, 3)

    # explicit num_steps=1 - caller asked for trajectory, gets time dim

    out = model(pos = pos, vel = vel, mass = mass, forces = forces, lens = lens, num_steps = 1, attrs = mass)

    assert out.pos.shape == (2, 1, 63, 3)
    assert out.vel.shape == (2, 1, 63, 3)

    # multi-step rollout

    out = model(pos = pos, vel = vel, mass = mass, forces = forces, lens = lens, num_steps = 3, attrs = mass)

    assert out.pos.shape == (2, 3, 63, 3)
    assert out.vel.shape == (2, 3, 63, 3)

    # without tokenizer - tokens passed directly

    model_no_tok = WorldParticle(predictor = predictor, corrector = corrector_kwargs)

    tokens = torch.randn(2, 63, 16)
    out = model_no_tok(tokens = tokens, pos = pos, vel = vel, mass = mass, forces = forces, lens = lens)

    assert out.pos.shape == (2, 63, 3)

    # test return_initial_state

    out = model_no_tok(tokens = tokens, pos = pos, vel = vel, mass = mass, forces = forces, lens = lens, return_initial_state = True)

    assert out.pos.shape == (2, 2, 63, 3) # single step inferred, but +1 for initial
    assert torch.allclose(out.pos[:, 0], pos)

    # 4D tokens auto-infers trajectory

    tokens_4d = torch.randn(2, 3, 63, 16)
    out = model_no_tok(tokens = tokens_4d, pos = pos, vel = vel, mass = mass, forces = forces, lens = lens)

    assert out.pos.shape == (2, 3, 63, 3)

    # 4D forces auto-infers trajectory

    forces_4d = torch.randn(2, 3, 63, 3)
    out = model(pos = pos, vel = vel, mass = mass, forces = forces_4d, lens = lens, attrs = mass)

    assert out.pos.shape == (2, 3, 63, 3)

@pytest.mark.skipif(not HAS_TORCH_CLUSTER, reason = 'torch-cluster is required for dynamic neighbor derivation')
def test_dynamic_neighbor_derivation():
    b, n = 2, 10
    dim = 16

    tokenizer = ParticleTokenizer(
        dim=dim,
        dim_attr=1,
        grid_res=5,
        spatial_radius=2.0,
        boundary_radius=2.0,
        max_spatial_neighbors=5,
        max_boundary_neighbors=4
    )

    # Dense tightly packed particles
    pos = torch.randn(b, n, 3)
    vel = torch.randn(b, n, 3)
    attrs = torch.randn(b, n, 1)

    boundary_pos = torch.randn(b, 15, 3)
    boundary_attrs = torch.randn(b, 15, 1)

    # Provide no indices!
    tokens = tokenizer(
        pos=pos,
        vel=vel,
        attrs=attrs,
        boundary_pos=boundary_pos,
        boundary_attrs=boundary_attrs
    )

    assert tokens.shape == (b, n, dim)

    # Ensure gradients flow
    loss = tokens.mean()
    loss.backward()
