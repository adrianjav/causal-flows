r"""Causal Normalizing Flow class."""

__all__ = ['CausalFlow']

from .distributions import CausalNormalizingFlow
from torch import Tensor
from zuko.flows import Flow


class CausalFlow(Flow):
    r"""Creates a lazy causal normalizing flow.

    See also:
        :class:`zuko.flows.Flow`
        :class:`causalflows.distributions.CausalNormalizingFlow`

    Arguments:
        transform: A lazy transformation or sequence of lazy transformations.
        base: A lazy distribution.
    """

    def forward(self, c: Tensor = None) -> CausalNormalizingFlow:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A normalizing flow :math:`p(X | c)`.
        """

        nflow = super().forward(c)
        return CausalNormalizingFlow(nflow.transform, nflow.base)
