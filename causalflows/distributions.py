r"""Causal Normalizing Flow distribution."""

__all__ = ['CausalNormalizingFlow']

import torch

from contextlib import contextmanager
from torch import LongTensor, Size, Tensor
from torch.distributions import Distribution, Transform
from typing import List
from zuko.distributions import NormalizingFlow

empty_size = Size()


class IntervenedTransform(Transform):
    def __init__(self, transform: Transform, index: List[LongTensor], value: List[Tensor]):
        super().__init__()
        self.transform = transform
        self.index: List[LongTensor] = index
        self.value: List[Tensor] = value

    def _inv_call(self, u):
        index = torch.cat(self.index, dim=-1)
        value = torch.stack(self.value, dim=-1)

        x = self.transform.inv(u)
        x[..., index] = value.to(device=x.device)
        u_tmp = self.transform(x)
        u[..., index] = u_tmp[..., index]
        return self.transform.inv(u)

    def __getattr__(self, item):
        return self.transform.__getattribute__(item)


class CausalNormalizingFlow(NormalizingFlow):
    r"""Class that extends :class:`zuko.distributions.NormalizingFlow` with
    methods to compute interventions and counterfactuals.

    Arguments:
        transform: A transformation :math:`f`.
        base: A base distribution :math:`p(Z)`.

    See also:
        - :class:`~zuko.distributions.NormalizingFlow` The equivalent non-causal counterpart from Zuko.

    Example:
        >>> d = CausalNormalizingFlow(ExpTransform(), Gamma(2.0, 1.0))
        >>> d.sample()
        tensor(1.5157)

    References:
        | A Family of Non-parametric Density Estimation Algorithms (Tabak et al., 2013)
        | https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.21423

        | Variational Inference with Normalizing Flows (Rezende et al., 2015)
        | https://arxiv.org/abs/1505.05770

        | Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios et al., 2021)
        | https://arxiv.org/abs/1912.02762

    """

    def __init__(
        self,
        transform: Transform,
        base: Distribution,
    ):
        super().__init__(transform, base)
        self.og_transform = transform
        self.indexes: List[LongTensor] = []
        self.values: List[Tensor] = []

    @contextmanager
    def intervene(
        self, index: LongTensor, value: Tensor
    ) -> NormalizingFlow:  # TODO CausalNormalizingFlow?
        r"""
        Context manager that yields an interventional distribution.

        Arguments:
            index: Index tensor of the intervened variables.
            value: Values of the intervened variables.

        Returns:
            A :class:`CausalNormalizingFlow` representing the
            interventional distribution.

        Warning:
            Nested interventions have not yet been tested.

        Example:
            >>> nflow = CausalNormalizingFlow(ExpTransform(), Gamma(2.0, torch.ones((1,))))
            >>> with nflow.intervene(index=0, value=0.5) as int_nflow:
            ...   x = int_nflow.sample((3,))
            >>> x
            tensor([[0.5000],
                    [0.5000],
                    [0.5000]])
        """
        try:
            yield self._start_intervention(index, value)
        except Exception as e:
            raise e
        finally:
            self._stop_intervention(index)

    def _start_intervention(self, index: LongTensor, value: Tensor) -> NormalizingFlow:
        if not torch.is_tensor(index):
            index = torch.tensor(index).long().view(-1)

        if not torch.is_tensor(value):
            value = torch.tensor(value)

        self.indexes.append(index)
        self.values.append(value)

        self.transform = IntervenedTransform(self.og_transform, self.indexes, self.values)

        return self

    def _stop_intervention(self, index: LongTensor) -> None:
        self.indexes.pop()
        self.values.pop()
        if len(self.indexes) == 0:
            self.transform = self.og_transform

    def sample_interventional(
        self, index: LongTensor, value: Tensor, sample_shape: Size = empty_size
    ) -> Tensor:
        r"""
        Helper method to sample from an interventional distribution.

        Arguments:
            index: Index tensor of the intervened variables.
            value: Values of the intervened variables.
            sample_shape: Batch shape of the samples.

        Returns:
             The intervened samples.

        Example:
            >>> nflow = CausalNormalizingFlow(ExpTransform(), Gamma(2.0, torch.ones((2,))))
            >>> x = nflow.sample_interventional(index=1, value=0.5, sample_shape=(3,))
            >>> x
            tensor([[ 1.5157,  0.5000],
                    [-0.4748,  0.5000],
                    [-0.1055,  0.5000]])
        """
        with self.intervene(index, value) as dist:
            return dist.sample(sample_shape)

    def compute_counterfactual(
        self,
        factual: Tensor,
        index: LongTensor,
        value: Tensor,
    ) -> Tensor:
        r"""
        Helper method to sample from a counterfactual distribution.

        Arguments:
            factual: The factual sample.
            index: Index tensor of the intervened variables.
            value: Values of the intervened variables.

        Returns:
             The counterfactual samples, with identical shape as :attr:`factual`.

        Example:
            >>> nflow = CausalNormalizingFlow(ExpTransform(), Gamma(2.0, torch.ones((2,))))
            >>> factual = nflow.sample((3,))
            >>> factual
            tensor([[ 1.5157,  0.2745],
                    [-0.4748, -0.8333],
                    [-0.1055,  0.1809]])
            >>> cfactual = nflow.compute_counterfactual(factual, index=0, value=0.5)
            >>> cfactual
            tensor([[ 0.5000,  0.2745],
                    [ 0.5000, -0.8333],
                    [ 0.5000,  0.1809]])
        """
        u = self.transform(factual)

        with self.intervene(index, value) as nflow:
            return nflow.transform.inv(u)
