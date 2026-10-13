import torch

from moshi.utils.sampling import sample_token


def test_sample_token_bias_promotes_selected_token():
    logits = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    token = sample_token(
        logits,
        use_sampling=True,
        temp=1.0,
        top_k=1,
        bias_token_id=2,
        bias=2.0,
    )

    assert token.item() == 2


def test_sample_token_bias_demotes_selected_token():
    logits = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    token = sample_token(
        logits,
        use_sampling=True,
        temp=1.0,
        top_k=1,
        bias_token_id=3,
        bias=-2.0,
    )

    assert token.item() != 3
