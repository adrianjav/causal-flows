Causal Normalizing Flows
========================


Causalflows is a Python package that implements causal normalizing flows in `PyTorch <https://pytorch.org>`_.
As of now, it is essentially a wrapper of the `Zuko <https://github.com/probabilists/zuko>`_ library with a number
of quality of life changes to improve its usability.


Citation
--------

To cite this library, please cite the original manuscript that preceded it:

.. code-block:: bibtex

    @article{javaloy2024causal,
      title={Causal normalizing flows: from theory to practice},
      author={Javaloy, Adri{\'a}n and S{\'a}nchez-Mart{\'\i}n, Pablo and Valera, Isabel},
      journal={Advances in {Neural} {Information} {Processing} {Systems}},
      volume={36},
      year={2024}
    }



Installation
------------

The :mod:`causalflows` package is still not publicly available, so you need to install it locally from the source folder
of this repository using

.. code-block:: console

    pip install -e .

Alternatively, you can install it directly from the repository.

.. code-block:: console

    pip install git+https://github.com/adrianjav/causal-flows

Getting started
---------------

Normalizing flows are provided in the :mod:`~causalflows.flows` module. To build one, supply the number of sample and
context features as well as the transformations' hyperparameters. Then, feeding a context :math:`c` to the flow returns
a conditional distribution :math:`p(x | c)` which can be evaluated and sampled from.

.. code-block:: python

    import torch
    import causalflows

    # Neural spline flow (NSF) with 3 sample features and 5 context features
    flow = causalflows.flows.CausalNSF(3, 5, order=(0, 1, 2), hidden_features=[128] * 3)

    # Train to maximize the log-likelihood
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    for x, c in trainset:
        loss = -flow(c).log_prob(x)  # -log p(x | c)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Sample 64 points x ~ p(x | c*)
    x = flow(c_star).sample((64,))

Alternatively, flows can be built as custom :class:`~causalflows.core.CausalFlow` objects.
As it can be appreciated in the snippet below, the library can be easily combined with custom flows
from the `Zuko <https://github.com/probabilists/zuko>`_ library.

.. code-block:: python

    from causalflows.flows import CausalFlow
    from zuko.flows import UnconditionalDistribution, UnconditionalTransform
    from zuko.flows.autoregressive import MaskedAutoregressiveTransform
    from zuko.distributions import DiagNormal
    from zuko.transforms import RotationTransform

    flow = CausalFlow(
        transform=[
            MaskedAutoregressiveTransform(3, 5, hidden_features=(64, 64)),
            UnconditionalTransform(RotationTransform, torch.randn(3, 3)),
            MaskedAutoregressiveTransform(3, 5, hidden_features=(64, 64)),
        ],
        base=UnconditionalDistribution(
            DiagNormal,
            torch.zeros(3),
            torch.ones(3),
            buffer=True,
        ),
    )

For more information, check out the :doc:`tutorials <../tutorials>` or the :doc:`API <../api>`.

References
----------

| Causal normalizing flows: from theory to practice (Javaloy et al., 2024)
| https://arxiv.org/abs/2306.05415

| NICE: Non-linear Independent Components Estimation (Dinh et al., 2014)
| https://arxiv.org/abs/1410.8516

| Variational Inference with Normalizing Flows (Rezende et al., 2015)
| https://arxiv.org/abs/1505.05770

| Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
| https://arxiv.org/abs/1705.07057

| Neural Spline Flows (Durkan et al., 2019)
| https://arxiv.org/abs/1906.04032

| Neural Autoregressive Flows (Huang et al., 2018)
| https://arxiv.org/abs/1804.00779

.. toctree::
    :caption: causalflows
    :hidden:
    :maxdepth: 2

    tutorials.rst
    api.rst

.. toctree::
    :caption: Development
    :hidden:
    :maxdepth: 1

    License <https://github.com/adrianjav/causal-flows/blob/master/LICENSE>
