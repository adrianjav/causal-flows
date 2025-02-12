r"""Doctests configuration."""

import causalflows
import pytest
import torch
import zuko


@pytest.fixture(autouse=True, scope="module")
def doctest_imports(doctest_namespace):
    doctest_namespace["torch"] = torch
    doctest_namespace["zuko"] = zuko
    doctest_namespace["causalflows"] = causalflows


@pytest.fixture(autouse=True)
def torch_seed():
    with torch.random.fork_rng():
        yield torch.random.manual_seed(0)
