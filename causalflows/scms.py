r"""Helper class to create synthetic SCMs and pre-defined instances of SCMs ready to use."""

from collections.abc import Callable, Sequence
from math import sqrt
from typing import Self, cast

import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor
from torch.distributions import Distribution, Independent, Normal, Transform, Uniform, constraints

from .distributions import CausalNormalizingFlow

NTen2Ten = (  # TODO type statement
    Callable[[Tensor], Tensor]
    | Callable[[Tensor, Tensor], Tensor]
    | Callable[[Tensor, Tensor, Tensor], Tensor]
    | Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
    | Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
)
Tens2Ten = Callable[[*tuple[Tensor, ...]], Tensor]  # TODO type statement


class CausalEquations(Transform):
    domain = constraints.unit_interval
    codomain = constraints.real
    bijective = True

    def __init__(
        self,
        functions: Sequence[Tens2Ten | NTen2Ten],
        inverses: Sequence[Tens2Ten | NTen2Ten],
        derivatives: Sequence[Tens2Ten | NTen2Ten] | None = None,
    ) -> None:
        super().__init__(cache_size=0)
        # Must use force cast because they are incompatible
        self.functions: Sequence[Tens2Ten] = cast(Sequence[Tens2Ten], functions)
        self.inverses: Sequence[Tens2Ten] = cast(Sequence[Tens2Ten], inverses)
        self.derivatives: Sequence[Tens2Ten] | None = cast(Sequence[Tens2Ten] | None, derivatives)

        self._interventions: dict[int, Tensor] = {}

    @property
    def adjacency(self) -> BoolTensor:
        raise NotImplementedError

    def _call(self, u: Tensor) -> Tensor:
        assert u.shape[1] == len(self.functions)

        x: list[Tensor] = []
        for i, f in enumerate(self.functions):
            if i in self._interventions:
                x_i = torch.ones_like(u[..., i]) * self._interventions[i]
            else:
                x_i = f(*x[:i], u[..., i])
            x.append(x_i)

        return torch.stack(x, dim=1)

    def _inverse(self, x: Tensor) -> Tensor:
        assert x.shape[1] == len(self.inverses)

        u: list[Tensor] = []
        for i, g in enumerate(self.inverses):
            u_i = g(*x[..., : i + 1].unbind(dim=-1))
            u.append(u_i)

        return torch.stack(u, dim=1)

    def log_abs_det_jacobian(self, u: Tensor, x: Tensor) -> Tensor:
        if self.derivatives is None:
            return self._log_abs_det_jacobian_autodiff(u, x)

        logdetjac: list[Tensor] = []
        for i, g in enumerate(self.derivatives):
            grad_i = g(*x[..., : i + 1].unbind(dim=-1))
            logdetjac.append(torch.log(grad_i.abs()))

        return -torch.stack(logdetjac, dim=-1)

    def _log_abs_det_jacobian_autodiff(self, u: Tensor, x: Tensor) -> Tensor:
        logdetjac: list[Tensor] = []
        old_requires_grad = x.requires_grad
        x.requires_grad_(True)
        for i, g in enumerate(self.inverses):  # u = T(x)
            u_i = g(*x[..., : i + 1].unbind(dim=-1))
            grad_i = torch.autograd.grad(u_i.sum(), x)[0][..., i]
            logdetjac.append(torch.log(grad_i.abs()))
        x.requires_grad_(old_requires_grad)
        return -torch.stack(logdetjac, dim=-1)

    def add_intervention(self, index: int, value: Tensor) -> None:
        self._interventions[index] = value

    def remove_intervention(self, index: int) -> None:
        self._interventions.pop(index)


class SCM(CausalNormalizingFlow):
    def __init__(self, equations: CausalEquations, base: str | Distribution) -> None:
        if isinstance(base, str):
            features = equations.adjacency.shape[0]
            if base == 'std-gaussian':
                base = Independent(
                    Normal(torch.zeros(features), torch.ones(features)),
                    reinterpreted_batch_ndims=1,
                )
            elif base == 'std-normal':
                base = Independent(
                    Uniform(torch.zeros(features), torch.ones(features)),
                    reinterpreted_batch_ndims=1,
                )
            else:
                raise ValueError(f'Unknown base distribution {base}.')

        self.equations = equations
        super().__init__(equations.inv, base)

    @property
    def adjacency(self) -> BoolTensor:
        adj = self.equations.adjacency
        assert (adj.shape[0] == adj.shape[1]) and (len(adj.shape) == 2)
        return adj

    def _start_intervention(self, index: int, value: Tensor | float | Sequence[float]) -> Self:
        self.equations.add_intervention(index, torch.as_tensor(value))
        return self

    def _stop_intervention(self, index: int) -> None:
        self.equations.remove_intervention(index)


####
# Examples
####


class Triangle(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        if eq_type == "linear":
            functions: list[NTen2Ten] = [
                lambda u1: u1 + 1.0,
                lambda x1, u2: 10 * x1 - u2,
                lambda x1, x2, u3: 0.5 * x2 + x1 + 1.0 * u3,
            ]
            inverses: list[NTen2Ten] = [
                lambda x1: x1 - 1.0,
                lambda x1, x2: (10 * x1 - x2),
                lambda x1, x2, x3: (x3 - 0.5 * x2 - x1) / 1.0,
            ]
        elif eq_type == 'non-linear':
            functions = [
                lambda u1: u1,
                lambda x1, u2: 2 * x1**2 + u2,
                lambda x1, x2, u3: 20.0 / (1 + torch.exp(-(x2**2) + x1)) + u3,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - 2 * x1**2,
                lambda x1, x2, x3: x3 - 20.0 / (1 + torch.exp(-(x2**2) + x1)),
            ]
        elif eq_type == 'non-linear-2':
            functions = [
                lambda u1: torch.sigmoid(u1),
                lambda x1, u2: 10 * x1**0.5 - u2,
                lambda x1, x2, u3: 0.5 * x2 + 1 / (1.0 + x1) + 1.0 * u3,
            ]
            inverses = [
                lambda x1: torch.logit(x1),
                lambda x1, x2: (10 * x1**0.5 - x2),
                lambda x1, x2, x3: (x3 - 0.5 * x2 - +1 / (1.0 + x1)) / 1.0,
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        return BoolTensor((
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
        ))


class Simpson(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        s = torch.nn.functional.softplus

        if eq_type == "non-linear":
            functions: list[NTen2Ten] = [
                lambda u1: u1,
                lambda x1, u2: s(1.0 - x1) + sqrt(3 / 20.0) * u2,
                lambda x1, x2, u3: torch.tanh(2 * x2) + 3 / 2 * x1 - 1 + torch.tanh(u3),
                lambda _1, _2, x3, u4: (x3 - 4.0) / 5.0 + 3 + 1 / sqrt(10) * u4,
            ]
            inverses: list[NTen2Ten] = [
                lambda x1: x1,
                lambda x1, x2: 1 / sqrt(3 / 20.0) * (x2 - s(1.0 - x1)),
                lambda x1, x2, x3: torch.atanh(x3 - torch.tanh(2 * x2) - 3 / 2 * x1 + 1),
                lambda _1, _2, x3, x4: sqrt(10) * (x4 - (x3 - 4.0) / 5.0 - 3),
            ]
        elif eq_type == 'sym-prod':
            functions = [
                lambda u1: u1,
                lambda x1, u2: 2 * torch.tanh(2 * x1) + 1 / sqrt(10) * u2,
                lambda x1, x2, u3: 1 / 2 * x1 * x2 + 1 / sqrt(2) * u3,
                lambda x1, _2, _3, u4: torch.tanh(3 / 2 * x1) + sqrt(3 / 10) * u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: sqrt(10) * (x2 - 2 * torch.tanh(2 * x1)),
                lambda x1, x2, x3: sqrt(2) * (x3 - 1 / 2 * x1 * x2),
                lambda x1, _2, _3, x4: 1 / sqrt(3 / 10) * (x4 - torch.tanh(3 / 2 * x1)),
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        if self.eq_type == "non-linear":
            return BoolTensor((
                (1, 0, 0, 0),
                (1, 1, 0, 0),
                (1, 1, 1, 0),
                (0, 0, 1, 1),
            ))
        if self.eq_type == "sym-prod":
            return BoolTensor((
                (1, 0, 0, 0),
                (1, 1, 0, 0),
                (1, 1, 1, 0),
                (1, 0, 0, 1),
            ))

        raise ValueError(f'Equation type {self.eq_type} not supported.')


class LargeBackdoor(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        def inv_softplus(bias: Tensor) -> Tensor:
            return bias.expm1().clamp_min(1e-6).log()

        def layer(x: Tensor, y: Tensor) -> Tensor:
            return F.softplus(x + 1) + F.softplus(0.5 + y) - 3.0

        def inv_layer(x: Tensor, z: Tensor) -> Tensor:
            return inv_softplus(z + 3 - F.softplus(x + 1)) - 0.5

        def icdf_laplace(loc: Tensor, scale: Tensor | float, value: Tensor) -> Tensor:
            term = value - 0.5
            return loc - scale * term.sign() * torch.log1p(-2 * term.abs())

        def cdf_laplace(loc: Tensor, scale: Tensor | float, value: Tensor) -> Tensor:
            return 0.5 - 0.5 * (value - loc).sign() * torch.expm1(-(value - loc).abs() / scale)

        if eq_type == "non-linear":
            functions: list[Tens2Ten] = [
                lambda *args: F.softplus(1.8 * args[-1]) - 1,  # x1
                lambda *args: 0.25 * args[-1]
                + layer(args[0], torch.zeros_like(args[-1])) * 1.5,  # x2
                lambda *args: layer(args[0], args[-1]),  # x3
                lambda *args: layer(args[1], args[-1]),  # x4
                lambda *args: layer(args[2], args[-1]),  # x5
                lambda *args: layer(args[3], args[-1]),  # x6
                lambda *args: layer(args[4], args[-1]),  # x7
                lambda *args: 0.3 * args[-1] + (F.softplus(args[5] + 1) - 1),  # x8
                lambda *args: icdf_laplace(
                    -F.softplus((args[6] * 1.3 + args[7]) / 3 + 1) + 2, 0.6, args[-1]
                ),  # x9
            ]
            inverses: list[Tens2Ten] = [
                lambda *args: inv_softplus(args[-1] + 1) / 1.8,  # u1
                lambda *args: 4
                * (-layer(args[0], torch.zeros_like(args[-1])) * 1.5 + args[-1]),  # u2
                lambda *args: inv_layer(args[0], args[-1]),  # u3
                lambda *args: inv_layer(args[1], args[-1]),  # u4
                lambda *args: inv_layer(args[2], args[-1]),  # u5
                lambda *args: inv_layer(args[3], args[-1]),  # u6
                lambda *args: inv_layer(args[4], args[-1]),  # u7
                lambda *args: (args[-1] - F.softplus(args[5] + 1) + 1) / 0.3,  # u8
                lambda *args: cdf_laplace(
                    -F.softplus((args[6] * 1.3 + args[7]) / 3 + 1) + 2, 0.6, args[-1]
                ),  # u9
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        return BoolTensor((
            (1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 1, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 1),
        ))


class Fork(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        if eq_type == "linear":
            functions: list[NTen2Ten] = [
                lambda u1: u1,
                lambda _1, u2: 2.0 - u2,
                lambda x1, x2, u3: 0.25 * x2 - 1.5 * x1 + 0.5 * u3,
                lambda _1, _2, x3, u4: 1.0 * x3 + 0.25 * u4,
            ]
            inverses: list[NTen2Ten] = [
                lambda x1: x1,
                lambda _1, x2: 2.0 - x2,
                lambda x1, x2, x3: (x3 - 0.25 * x2 + 1.5 * x1) / 0.5,
                lambda _1, _2, x3, x4: (x4 - 1.0 * x3) / 0.25,
            ]
        elif eq_type == 'non-linear':
            functions = [
                lambda u1: u1,
                lambda _1, u2: u2,
                lambda x1, x2, u3: 4.0 / (1 + torch.exp(-x1 - x2)) - x2**2 + u3 / 2.0,
                lambda _1, _2, x3, u4: 20.0 / (1 + torch.exp(x3**2 / 2 - x3)) + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda _1, x2: x2,
                lambda x1, x2, x3: 2.0 * (x3 - 4.0 / (1 + torch.exp(-x1 - x2)) + x2**2),
                lambda _1, _2, x3, x4: x4 - 20.0 / (1 + torch.exp(x3**2 / 2 - x3)),
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        return BoolTensor((
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (1, 1, 1, 0),
            (0, 0, 1, 1),
        ))


class Diamond(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        if eq_type == "non-linear":
            functions: list[NTen2Ten] = [
                lambda u1: u1,
                lambda x1, u2: x1**2 + u2 / 2,
                lambda x1, x2, u3: x2**2 - 2.0 / (1 + torch.exp(-x1)) + u3 / 2.0,
                lambda _1, x2, x3, u4: x3 / ((x2 + 2.0).abs() + x3 + 0.5) + u4 / 10.0,
            ]
            inverses: list[NTen2Ten] = [
                lambda x1: x1,
                lambda x1, x2: 2 * (x2 - x1**2),
                lambda x1, x2, x3: (x3 - x2**2 + 2.0 / (1 + torch.exp(-x1))) * 2.0,
                lambda _1, x2, x3, x4: 10 * (x4 - x3 / ((x2 + 2.0).abs() + x3 + 0.5)),
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        return BoolTensor((
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 1, 0),
            (0, 1, 1, 1),
        ))


class Collider(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        if eq_type == "linear":
            functions: list[NTen2Ten] = [
                lambda u1: u1,
                lambda _1, u2: 2.0 - u2,
                lambda x1, x2, u3: 0.25 * x2 - 0.5 * x1 + 0.5 * u3,
            ]
            inverses: list[NTen2Ten] = [
                lambda x1: x1,
                lambda _1, x2: 2.0 - x2,
                lambda x1, x2, x3: (x3 - 0.25 * x2 + 0.5 * x1) / 0.5,
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        return BoolTensor((
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 1),
        ))


class Chain3(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        if eq_type == "linear":
            functions: list[NTen2Ten] = [
                lambda u1: u1,
                lambda x1, u2: 10 * x1 - u2,
                lambda _1, x2, u3: 0.25 * x2 + 2 * u3,
            ]
            inverses: list[NTen2Ten] = [
                lambda x1: x1,
                lambda x1, x2: (10 * x1 - x2),
                lambda _1, x2, x3: (x3 - 0.25 * x2) / 2,
            ]
        elif eq_type == 'non-linear':
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.exp(x1 / 2.0) + u2 / 4.0,
                lambda _1, x2, u3: (x2 - 5) ** 3 / 15.0 + u3,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: 4.0 * (x2 - torch.exp(x1 / 2.0)),
                lambda _1, x2, x3: x3 - (x2 - 5) ** 3 / 15.0,
            ]
        elif eq_type == 'non-linear-2':
            functions = [
                lambda u1: torch.sigmoid(u1),
                lambda x1, u2: 10 * x1**0.5 - u2,
                lambda _1, x2, u3: 0.25 * x2 + 2 * u3,
            ]
            inverses = [
                lambda x1: torch.logit(x1),
                lambda x1, x2: (10 * x1**0.5 - x2),
                lambda _1, x2, x3: (x3 - 0.25 * x2) / 2,
            ]
        elif eq_type == 'non-linear-3':
            functions = [
                lambda u1: u1,
                lambda x1, u2: 1 * x1**2.0 - u2,
                lambda _1, x2, u3: 0.25 * x2 + 2 * u3,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: (1 * x1**2.0 - x2),
                lambda _1, x2, x3: (x3 - 0.25 * x2) / 2,
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        return BoolTensor((
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 1),
        ))


class Chain4(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        if eq_type == "linear":
            functions: list[NTen2Ten] = [
                lambda u1: u1,
                lambda x1, u2: 5 * x1 - u2,
                lambda _1, x2, u3: -0.5 * x2 - 1.5 * u3,
                lambda _1, _2, x3, u4: x3 + u4,
            ]
            inverses: list[NTen2Ten] = [
                lambda x1: x1,
                lambda x1, x2: (5 * x1 - x2),
                lambda _1, x2, x3: (-0.5 * x2 - x3) / 1.5,
                lambda _1, _2, x3, x4: x4 - x3,
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        return BoolTensor((
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (0, 0, 1, 1),
        ))


class Chain5(CausalEquations):
    def __init__(self, eq_type: str) -> None:
        if eq_type == "linear":
            functions: list[NTen2Ten] = [
                lambda u1: u1,
                lambda x1, u2: 10 * x1 - u2,
                lambda _1, x2, u3: 0.25 * x2 + 2 * u3,
                lambda _1, _2, x3, u4: x3 + u4,
                lambda _1, _2, _3, x4, u5: -x4 + u5,
            ]
            inverses: list[NTen2Ten] = [
                lambda x1: x1,
                lambda x1, x2: (10 * x1 - x2),
                lambda _1, x2, x3: (x3 - 0.25 * x2) / 2,
                lambda _1, _2, x3, x4: x4 - x3,
                lambda _1, _2, _3, x4, x5: x5 + x4,
            ]
        else:
            raise ValueError(f'Equation type {eq_type} not supported.')

        self.eq_type = eq_type
        super().__init__(functions, inverses)

    @property
    def adjacency(self) -> BoolTensor:
        return BoolTensor((
            (1, 0, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (0, 1, 1, 0, 0),
            (0, 0, 1, 1, 0),
            (0, 0, 0, 1, 1),
        ))
