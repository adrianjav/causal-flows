import torch

from .distributions import CausalNormalizingFlow
from torch import Tensor
from typing import Sequence, Union
from zuko.flows import Flow, LazyDistribution, LazyTransform

__all__ = ["CausalFlow"]


class CausalFlow(Flow):
    r"""Creates a lazy causal normalizing flow.

    See also:
        :class:`zuko.flows.Flow`
        :class:`causalflows.distributions.CausalNormalizingFlow`

    Arguments:
        transform: A lazy transformation or sequence of lazy transformations.
        base: A lazy distribution.
    """

    def __init__(
        self,
        transform: Union[LazyTransform, Sequence[LazyTransform]],
        base: LazyDistribution,
    ):
        super().__init__(transform, base)
        self.register_buffer("intervention_index", torch.empty([]))
        self.register_buffer("intervention_value", torch.empty([]))

    def forward(self, c: Tensor = None) -> CausalNormalizingFlow:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A normalizing flow :math:`p(X | c)`.
        """

        nflow = super().forward(c)
        return CausalNormalizingFlow(nflow.transform, nflow.base)
