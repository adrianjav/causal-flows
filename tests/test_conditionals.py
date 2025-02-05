import torch

from causalflows.flows import CausalMAF


def test_conditionals():
    def allclose(x, v):
        return torch.allclose(x, torch.ones_like(x) * v)

    features = 4
    context = 2
    nsamples = 5

    flow = CausalMAF(features, context, order=range(features), hidden_features=[16] * 3)
    nflow = flow(torch.randn((nsamples, context)))

    # Important! Here we sample 1 sample per context, which there are nsamples of them, so samples are have shape (1, nsamples, features)
    assert nflow.sample((1,)).shape == (1, nsamples, features)

    with nflow.intervene(1, 0.0):
        x_int = nflow.sample((1,)).squeeze(0)
        print(x_int)
        assert x_int.shape == (nsamples, features)
        assert allclose(x_int[:, 1], 0.0)

    with nflow.intervene(1, 0.0):
        with nflow.intervene(2, 1.0):
            x_int = nflow.sample((1,)).squeeze(0)
            print(x_int)
            assert x_int.shape == (nsamples, features)
            assert allclose(x_int[:, 2], 1.0) and allclose(x_int[:, 1], 0.0)

    x = nflow.sample((1,)).squeeze(0)
    assert not (allclose(x[:, 2], 1.0) and allclose(x[:, 1], 0.0))

    x_int = nflow.sample_interventional(index=1, value=5.0, sample_shape=(1,)).squeeze(0)
    assert x_int.shape == (nsamples, features)
    assert allclose(x_int[:, 1], 5.0)

    x_cf = nflow.compute_counterfactual(x, index=2, value=2.0).squeeze(0)
    assert x_cf.shape == (nsamples, features)
    assert (
        torch.allclose(x_cf[:, :1], x[:, :1])
        and allclose(x_cf[:, 2], 2.0)
        and not torch.allclose(x_cf[:, 3:], x[:, 3:])
    )
