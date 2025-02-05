r"""Tests for the zuko.flows module."""

import pytest
import torch

from causalflows.flows import *
from math import pi
from pathlib import Path
from torch import randn


# @pytest.mark.parametrize("F", [CausalMAF, CausalNSF, CausalNAF, CausalUNAF, CausalNCSF])
@pytest.mark.parametrize("F", [CausalMAF, CausalNSF, CausalNAF, CausalUNAF, CausalNCSF])
def test_flows(tmp_path: Path, F: callable):
    flow = F(3, 5, order=(0, 1, 2))

    # Evaluation of log_prob
    x, c = randn(256, 3), randn(5)
    if F is CausalNCSF:
        x = x.clamp(min=-pi + 1e-3, max=pi - 1e-3)
    log_p = flow(c).log_prob(x)

    assert log_p.shape == (256,)
    assert log_p.requires_grad

    flow.zero_grad(set_to_none=True)
    loss = -log_p.mean()
    loss.backward()

    for p in flow.parameters():
        assert p.grad is not None

    # Sampling
    x = flow(c).sample((32,))

    assert x.shape == (32, 3)

    # Reparameterization trick
    if flow(c).has_rsample:
        x = flow(c).rsample()

        flow.zero_grad(set_to_none=True)
        loss = x.square().sum().sqrt()
        loss.backward()

        for p in flow.parameters():
            assert p.grad is not None

    # Invertibility
    if isinstance(flow, CausalFlow):
        x, c = randn(256, 3), randn(256, 5)
        if F is CausalNCSF:
            x = x.clamp(min=-pi + 1e-3, max=pi - 1e-3)
        t = flow(c).transform
        z = t.inv(t(x))

        assert torch.allclose(x, z, atol=1e-4)

    # Saving
    torch.save(flow.state_dict(), tmp_path / "flow.pt")

    # Loading
    flow_state_dict = torch.load(tmp_path / "flow.pt")
    flow_bis = F(3, 5, order=(0, 1, 2))
    flow_bis.load_state_dict(flow_state_dict)

    x, c = randn(3), randn(5)
    if F is CausalNCSF:
        x = x.clamp(min=-pi + 1e-3, max=pi - 1e-3)

    seed = torch.seed()
    log_p = flow(c).log_prob(x)
    torch.manual_seed(seed)
    log_p_bis = flow_bis(c).log_prob(x)

    assert torch.allclose(log_p, log_p_bis)

    # Printing
    assert repr(flow)
