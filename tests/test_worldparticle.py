import torch
from worldparticle.worldparticle import merge_tokens

def test_merge_tokens():
    tokens = torch.randn(2, 63, 16)
    lens = torch.tensor([63, 31])
    weights = None

    for _ in range(10):
        tokens, weights, lens = merge_tokens(tokens, weights, lens)
        if (lens == 1).all():
            break

    assert (lens == 1).all()
    assert (weights[:, 0] >= torch.tensor([63., 31.])).all()
