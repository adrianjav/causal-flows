# Causal Normalizing Flows

> [!warning]
> This is work in progress. You can expect bugs (yet we do not know of any) and rough edges.

CausalFlows is a Python package that implements causal normalizing flows in [PyTorch](https://pytorch.org>).
As of now, it is essentially a wrapper of the [Zuko](https://github.com/probabilists/zuko) library with a number
of quality of life changes to improve its usability.

## Installation

The package is still not publicly available, so you need to install it locally from the source folder
of this repository using

```bash
pip install -e .
```
Alternatively, you can install it directly from the repository.

```bash
pip install git+https://github.com/adrianjav/causal-flows
```

## Getting started

Normalizing flows are provided in the [flows](causalflows/flows) module. To build one, supply the number of sample and
context features as well as the transformations' hyperparameters. Then, feeding a context $c$ to the flow returns
a conditional distribution $p(x | c)$ which can be evaluated and sampled from.

```python
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

# Sample 64 factual points x ~ p(x | c*)
x = flow(c_star).sample((64,))

# Intervene using the context manager (the context needs always to be given)
with flow(c_star).intervene(index=1, value=2.5) as int_flow:
    x_int = int_flow.sample((64,))

# We could also sample with the helper method
x_int = flow(c_star).sample_interventional(index=1, value=2.5, sample_shape=(64,))

# And we can compute counterfactuals using the helper methods (or the context manager)
x_cf = flow(c_star).compute_counterfactual(x, index=1, value=2.5)
```

Alternatively, flows can be built as custom [CausalFlow](https://github.com/adrianjav/causal-flows/blob/189e7d6ea35a4000b2899a2c54ed4883c58ffed9/causalflows/core.py#L11) objects.
As it can be appreciated in the snippet below, the library can be easily combined with custom flows
from the [Zuko](https://github.com/probabilists/zuko) library.

```python
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
```

For more information, check out the [tutorials](docs/tutorials) or the [documentation](docs).

## References

> Causal normalizing flows: from theory to practice (Javaloy et al., 2024)
> https://arxiv.org/abs/2306.05415
>
> NICE: Non-linear Independent Components Estimation (Dinh et al., 2014)
> https://arxiv.org/abs/1410.8516
>
> Variational Inference with Normalizing Flows (Rezende et al., 2015)
> https://arxiv.org/abs/1505.05770
>
> Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
> https://arxiv.org/abs/1705.07057
>
> Neural Spline Flows (Durkan et al., 2019)
> https://arxiv.org/abs/1906.04032
>
> Neural Autoregressive Flows (Huang et al., 2018)
> https://arxiv.org/abs/1804.00779
