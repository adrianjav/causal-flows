r"""Doctests configuration."""

import causalflows
import pytest
import torch
import zuko

from torch.distributions import ExpTransform, Gamma


@pytest.fixture(autouse=True, scope="module")
def doctest_imports(doctest_namespace):
    doctest_namespace["torch"] = torch
    doctest_namespace["zuko"] = zuko
    doctest_namespace["causalflows"] = causalflows

    # For the examples in the doctests
    doctest_namespace["ExpTransform"] = ExpTransform
    doctest_namespace["Gamma"] = Gamma


@pytest.fixture(autouse=True)
def torch_seed():
    with torch.random.fork_rng():
        yield torch.random.manual_seed(0)
