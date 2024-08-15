r"""Wrappers for causal normalizing flows using standard architectures."""

__all__ = ['CausalFlow', 'CausalMAF', 'CausalNAF', 'CausalNCSF', 'CausalNSF', 'CausalUNAF']

import torch

from .core import CausalFlow
from math import pi
from torch import BoolTensor, LongTensor, Size
from typing import (
    Any,
    Dict,
    Optional,
)
from zuko.distributions import BoxUniform, DiagNormal
from zuko.flows import (
    MaskedAutoregressiveTransform,
    UnconditionalDistribution,
)
from zuko.flows.neural import MNN, UMNN
from zuko.flows.spline import CircularRQSTransform
from zuko.transforms import MonotonicRQSTransform


class CausalMAF(CausalFlow):
    r"""Creates a causal flow using a masked autoregressive flow as base the base model (Causal MAF).

    See also:
        :class:`zuko.flows.autoregressive.MAF`

    References:
        | Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
        | https://arxiv.org/abs/1705.07057

    Arguments:
        features: The number of features.
        context: The number of context features.
        order: The causal order to follow by the flow. If used, then `adjacency` must be :py:`None`.
        adjacency: The causal graph to follow by the flow. If used, then `order` must be :py:`None`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MaskedAutoregressiveTransform`.

    Example:
        >>> flow = CausalMAF(3, 4, order=torch.arange(3))
        >>> flow
        CausalMAF(
          (transform): LazyInverse(
            (transform): MaskedAutoregressiveTransform(
              (base): MonotonicAffineTransform()
              (order): [0, 1, 2]
              (hyper): MaskedMLP(
                (0): MaskedLinear(in_features=7, out_features=64, bias=True)
                (1): ReLU()
                (2): MaskedLinear(in_features=64, out_features=64, bias=True)
                (3): ReLU()
                (4): MaskedLinear(in_features=64, out_features=6, bias=True)
              )
            )
          )
          (base): UnconditionalDistribution(DiagNormal(loc: torch.Size([3]), scale: torch.Size([3])))
        )
        >>> c = torch.randn(4)
        >>> x = flow(c).sample()
        >>> x
        tensor([-2.3677, -0.0753, -0.9235])
        >>> flow(c).log_prob(x)
        tensor(-5.6385, grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        *args,
        order: Optional[LongTensor] = None,
        adjacency: Optional[BoolTensor] = None,
        **kwargs,
    ):
        assert (order is None) != (
            adjacency is None
        ), "One of `order` or `adjacency` must be specified."

        transform = MaskedAutoregressiveTransform(
            features=features,
            context=context,
            order=order,
            adjacency=adjacency,
            **kwargs,
        )

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transform, base)


class CausalNSF(CausalMAF):
    r"""Creates a causal flow using a neural spline flow (NSF) with monotonic rational-quadratic spline
    transformations as base model.

    Warning:
        Spline transformations are defined over the domain :math:`[-5, 5]`. Any feature
        outside of this domain is not transformed. It is recommended to standardize
        features (zero mean, unit variance) before training.

    See also:
        :class:`zuko.flows.spline.NSF`
        :class:`zuko.transforms.CircularShiftTransform`

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`causalflows.CausalMAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )


class CausalNCSF(CausalMAF):
    r"""Creates a causal flow using a neural circular spline flow (NCSF) as base model.

    Circular spline transformations are obtained by composing circular domain shifts
    with regular spline transformations. Features are assumed to lie in the half-open
    interval :math:`[-\pi, \pi)`.

    See also:
        :class:`zuko.flows.spline.NSCF`
        :class:`zuko.transforms.CircularShiftTransform`

    References:
        | Normalizing Flows on Tori and Spheres (Rezende et al., 2020)
        | https://arxiv.org/abs/2002.02428

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=CircularRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.base = UnconditionalDistribution(
            BoxUniform,
            torch.full((features,), -pi - 1e-5),
            torch.full((features,), pi + 1e-5),
            buffer=True,
        )


class CausalNAF(CausalFlow):
    r"""Creates a causal flow using a neural autoregressive flow (NAF) as base model.

    See also:
        :class:`zuko.flows.neural.NAF`

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Neural Autoregressive Flows (Huang et al., 2018)
        | https://arxiv.org/abs/1804.00779

    Arguments:
        features: The number of features.
        context: The number of context features.
        signal: The number of signal features of the monotonic network.
        order: The causal order to follow by the flow. If used, then `adjacency` must be :py:`None`.
        adjacency: The causal graph to follow by the flow. If used, then `order` must be :py:`None`.
        network: Keyword arguments passed to :class:`zuko.flows.neural.MNN`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        signal: int = 16,
        *args,
        order: Optional[LongTensor] = None,
        adjacency: Optional[BoolTensor] = None,
        network: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        assert (order is None) != (
            adjacency is None
        ), "One of `order` or `adjacency` must be specified."

        if network is None:
            network = {}

        transforms = MaskedAutoregressiveTransform(
            features=features,
            context=context,
            order=order,
            adjacency=adjacency,
            univariate=MNN(signal=signal, stack=features, **network),
            shapes=[Size((signal,))],
            **kwargs,
        )

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class CausalUNAF(CausalFlow):
    r"""Creates a causal flow using an unconstrained neural autoregressive flow (UNAF) as base model.

    See also:
        :class:`zuko.flows.neural.UNAF`

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Unconstrained Monotonic Neural Networks (Wehenkel et al., 2019)
        | https://arxiv.org/abs/1908.05164

    Arguments:
        features: The number of features.
        context: The number of context features.
        signal: The number of signal features of the monotonic network.
        order: The causal order to follow by the flow. If used, then `adjacency` must be :py:`None`.
        adjacency: The causal graph to follow by the flow. If used, then `order` must be :py:`None`.
        network: Keyword arguments passed to :class:`zuko.flows.neural.UMNN`.
        kwargs: Keyword arguments passed to :class:`zuko.flows.autoregressive.MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        signal: int = 16,
        *args,
        order: Optional[LongTensor] = None,
        adjacency: Optional[BoolTensor] = None,
        network: Dict[str, Any] = None,
        **kwargs,
    ):
        assert (order is None) != (
            adjacency is None
        ), "One of `order` or `adjacency` must be specified."

        if network is None:
            network = {}

        transforms = MaskedAutoregressiveTransform(
            features=features,
            context=context,
            order=order,
            adjacency=adjacency,
            univariate=UMNN(signal=signal, stack=features, **network),
            shapes=[Size((signal,)), ()],
            **kwargs,
        )

        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)
