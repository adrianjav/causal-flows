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
    r"""Creates a causal flow using a masked autoregressive flow (MAF) as base model.

    Arguments:
        features: The number of features.
        context: The number of context features.
        order: The causal order to follow by the flow. If used, then `adjacency` must be :py:`None`.
        adjacency: The causal graph to pass to the flow. If used, then `order` must be :py:`None`.
        kwargs: Keyword arguments passed to :class:`~zuko.flows.autoregressive.MaskedAutoregressiveTransform`.

    See also:
        - :class:`~zuko.flows.autoregressive.MAF` The equivalent non-causal counterpart from Zuko.

    Example:
        >>> flow = CausalMAF(3, 4, order=torch.arange(3))
        >>> flow
        CausalMAF(
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
          (base): UnconditionalDistribution(DiagNormal(loc: torch.Size([3]), scale: torch.Size([3])))
        )
        >>> c = torch.randn(4)
        >>> x = flow(c).sample()
        >>> x
        tensor([-1.9301, -0.1411, -0.7982])
        >>> flow(c).log_prob(x)
        tensor(-5.1800, grad_fn=<AddBackward0>)

    References:
        | Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
        | https://arxiv.org/abs/1705.07057
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
    r"""Creates a causal flow using a neural spline flow (NSF) as the base model.

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`~causalflows.flows.CausalMAF`.

    See also:
        - :class:`~zuko.flows.spline.NSF` The equivalent non-causal counterpart from Zuko.

    Warning:
        Spline transformations are defined over the domain :math:`[-5, 5]`. Any feature
        outside of this domain is not transformed. It is recommended to standardize
        features (zero mean, unit variance) before training.


    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

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

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`~zuko.flows.autoregressive.MAF`.

    See also:
        - :class:`~zuko.flows.spline.NCSF` The equivalent non-causal counterpart from Zuko.

    References:
        | Normalizing Flows on Tori and Spheres (Rezende et al., 2020)
        | https://arxiv.org/abs/2002.02428
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

    Arguments:
        features: The number of features.
        context: The number of context features.
        signal: The number of signal features of the monotonic network.
        order: The causal order to follow by the flow. If used, then `adjacency` must be :py:`None`.
        adjacency: The causal graph to follow by the flow. If used, then `order` must be :py:`None`.
        network: Keyword arguments passed to :class:`~zuko.flows.neural.MNN`.
        kwargs: Keyword arguments passed to :class:`~zuko.flows.autoregressive.MaskedAutoregressiveTransform`.

    See also:
        - :class:`~zuko.flows.neural.NAF` The equivalent non-causal counterpart from Zuko.

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Neural Autoregressive Flows (Huang et al., 2018)
        | https://arxiv.org/abs/1804.00779

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

    Arguments:
        features: The number of features.
        context: The number of context features.
        signal: The number of signal features of the monotonic network.
        order: The causal order to follow by the flow. If used, then `adjacency` must be :py:`None`.
        adjacency: The causal graph to follow by the flow. If used, then `order` must be :py:`None`.
        network: Keyword arguments passed to :class:`~zuko.flows.neural.UMNN`.
        kwargs: Keyword arguments passed to :class:`~zuko.flows.autoregressive.MaskedAutoregressiveTransform`.

    See also:
        - :class:`~zuko.flows.neural.UNAF` The equivalent non-causal counterpart from Zuko.

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Unconstrained Monotonic Neural Networks (Wehenkel et al., 2019)
        | https://arxiv.org/abs/1908.05164

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
