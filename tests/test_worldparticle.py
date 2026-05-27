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

def test_merge_tokens_masked_target():
    # already 1 token

    tokens = torch.ones(1, 1, 16)
    tokens_out, weights_out, lens_out = merge_tokens(tokens, lens = torch.tensor([1]))

    assert (lens_out == 1).all()
    assert (weights_out == 1.).all()
    assert torch.allclose(tokens_out, tokens)

    # batch with masked out targets

    tokens = torch.ones(2, 3, 16)
    tokens_out, weights_out, lens_out = merge_tokens(tokens, weights = torch.ones(2, 3), lens = torch.tensor([1, 3]))

    assert lens_out[0] == 2
    assert weights_out[0, 0] == 1.
    assert weights_out[0, 1] == 0.
