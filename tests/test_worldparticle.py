import pytest
param = pytest.mark.parametrize

import torch
from worldparticle.worldparticle import merge_tokens, ParticleTransformerCorrector

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
