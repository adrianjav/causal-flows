import torch

from contextlib import contextmanager
from torch import LongTensor, Size, Tensor
from torch.distributions import Distribution, Transform
from typing import List
from zuko.distributions import NormalizingFlow

__all__ = ["CausalNormalizingFlow"]

empty_size = Size()


class IntervenedTransform(Transform):
    def __init__(self, transform: Transform, index: List[LongTensor], value: List[Tensor]):
        super().__init__()
        self.transform = transform
        self.index: List[LongTensor] = index
        self.value: List[Tensor] = value

    # TODO For now nested interventions need to be given in order
    def _inv_call(self, u):
        index = torch.cat(self.index, dim=-1)
        value = torch.stack(self.value, dim=-1)

        x = self.transform.inv(u)
        x[..., index] = value.to(device=x.device)
        u_tmp = self.transform(x)
        u_int = u.clone()  # Avoid aliasing
        u_int[..., index] = u_tmp[..., index]
        return self.transform.inv(u_int)

    def __getattr__(self, item):
        return self.transform.__getattribute__(item)


class CausalNormalizingFlow(NormalizingFlow):  # TODO Document new methods better
    r"""Class that extends :class:`~zuko.distributions.NormalizingFlow` with
    methods to compute interventions and counterfactuals.

    See also:
        :class:`~zuko.distributions.NormalizingFlow`

    References:
        | A Family of Non-parametric Density Estimation Algorithms (Tabak et al., 2013)
        | https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.21423

        | Variational Inference with Normalizing Flows (Rezende et al., 2015)
        | https://arxiv.org/abs/1505.05770

        | Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios et al., 2021)
        | https://arxiv.org/abs/1912.02762

    Arguments:
        transform: A transformation :math:`f`.
        base: A base distribution :math:`p(Z)`.

    Example:
        >>> d = CausalNormalizingFlow(ExpTransform(), Gamma(2.0, 1.0))
        >>> d.sample()
        tensor(1.5157)
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
            An :class:`~torch.distributions.Distribution` representing the
            interventional distribution.
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
        Samples from the interventional distribution.

        Arguments:
            index: Index tensor of the intervened variables.
            value: Values of the intervened variables.
            sample_shape: Batch shape of the samples.

        Returns:
             The interventional samples.
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
        Samples from the counterfactual distribution.

        Arguments:
            factual: The factual sample.
            index: Index tensor of the intervened variables.
            value: Values of the intervened variables.

        Returns:
             The counterfactual samples.
        """
        u = self.transform(factual)

        with self.intervene(index, value) as nflow:
            return nflow.transform.inv(u)
