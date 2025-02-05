import torch
import zuko

from causalflows.distributions import CausalNormalizingFlow


def test_interventions():
    def allclose(x, v):
        return torch.allclose(x, torch.ones_like(x) * v)

    features = 4
    context = 0

    transform = zuko.flows.LazyInverse(
        zuko.flows.MaskedAutoregressiveTransform(
            features=features,
            context=context,
        )
    )

    base = zuko.flows.UnconditionalDistribution(
        zuko.distributions.DiagNormal,
        torch.zeros(features),
        torch.ones(features),
        buffer=True,
    )

    nflow = CausalNormalizingFlow(transform(), base())

    assert nflow.sample((4,)).shape == (4, features)

    with nflow.intervene(1, 0.0):
        x_int = nflow.sample((4,))
        print(x_int)
        assert x_int.shape == (4, features)
        assert allclose(x_int[:, 1], 0.0)

    with nflow.intervene(1, 0.0):
        with nflow.intervene(2, 1.0):
            x_int = nflow.sample((4,))
            print(x_int)
            assert x_int.shape == (4, features)
            assert allclose(x_int[:, 2], 1.0) and allclose(x_int[:, 1], 0.0)

    x = nflow.sample((4,))
    assert not (allclose(x[:, 2], 1.0) and allclose(x[:, 1], 0.0))

    x_int = nflow.sample_interventional(index=1, value=5.0, sample_shape=(4,))
    assert x_int.shape == (4, features)
    assert allclose(x_int[:, 1], 5.0)

    x_cf = nflow.compute_counterfactual(x, index=2, value=2.0)
    assert x_cf.shape == (4, features)
    assert (
        torch.allclose(x_cf[:, :1], x[:, :1])
        and allclose(x_cf[:, 2], 2.0)
        and not torch.allclose(x_cf[:, 3:], x[:, 3:])
    )
