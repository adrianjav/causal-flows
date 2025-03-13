r"""Causal Normalizing Flow class."""

__all__ = ['CausalFlow']

from typing import Optional

from torch import Tensor
from zuko.flows import Flow

from .distributions import CausalNormalizingFlow


class CausalFlow(Flow):
    r"""Creates a lazy causal normalizing flow (initialized after passing the context).

    See also:
        - :class:`zuko.flows.core.Flow` - Flow base class from Zuko.
        - :class:`causalflows.distributions.CausalNormalizingFlow` - Class returned after the forward pass.

    Arguments:
        transform: A lazy transformation or sequence of lazy transformations.
        base: A lazy distribution.
    """

    def forward(self, context: Optional[Tensor] = None) -> CausalNormalizingFlow:
        r"""
        Arguments:
            context: An input tensor representing the context of the (conditional) flow.

        Returns:
            A causal normalizing flow :math:`p(X | c)`.
        """

        nflow = super().forward(context)
        return CausalNormalizingFlow(nflow.transform, nflow.base)
