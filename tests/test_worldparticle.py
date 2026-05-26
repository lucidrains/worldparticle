import pytest
import torch

def test_worldparticle():
    from worldparticle.worldparticle import merge_tokens

    tokens = torch.randn(2, 511, 64)
    lens = None
    weights = None

    for _ in range(1):
        tokens, weights, lens = merge_tokens(tokens, weights, lens)

    assert tokens.shape[1] == 256
    assert (lens >= 1).all()
