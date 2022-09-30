"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:

    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):

    def forward(self, x):
        return x


class Linear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(self.in_features, self.out_features)
        self.bias = init.kaiming_uniform(self.out_features, 1).reshape(
            (1, self.out_features)) if bias else None

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = X.shape[0]
        ret = ops.matmul(X, self.weight)
        if self.bias != None:
            ret += ops.broadcast_to(self.bias, ret.shape)
        return ret
        ### END YOUR SOLUTION


class Flatten(Module):

    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        n = X.shape[0]
        dim = 1
        for i in range(1, len(X.shape)):
            dim *= X.shape[i]
        return ops.reshape(X, (n, dim))
        ### END YOUR SOLUTION


class ReLU(Module):

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):

    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n = logits.shape[0]
        label = init.one_hot(n, y)
        x = ops.exp(logits).sum(1)
        y = ops.log(x).sum()
        z = (logits * label).sum()
        loss = y - z
        return loss / n
        ### END YOUR SOLUTION


class BatchNorm1d(Module):

    def __init__(self,
                 dim,
                 eps=1e-5,
                 momentum=0.1,
                 device=None,
                 dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Tensor(np.ones((1, dim)), device=device, dtype=dtype)
        self.bias = Tensor(np.zeros((1, dim)), device=device, dtype=dtype)
        self.running_mean = Tensor(np.zeros((dim)), device=device, dtype=dtype)
        self.running_var = Tensor(np.ones((dim)), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # self.bias = self.bias.reshape((1, self.dim))
        # self.weight = self.weight.reshape((1, self.dim))
        n, n_feature = x.shape[0], x.shape[1]
        if self.training:
            m = (x.sum(0) / n)
            mean = ops.broadcast_to(m, x.shape)
            var = ((x - mean)**2).sum(0) / n
            ret = (x - mean) / ops.broadcast_to(
                ops.power_scalar(var + self.eps, 0.5), x.shape
            ) * ops.broadcast_to(self.weight, x.shape) + ops.broadcast_to(
                self.bias, x.shape)
            self.running_mean = self.momentum * m + (
                1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (
                1 - self.momentum) * self.running_var
            return ret
        else:
            mean = ops.broadcast_to(self.running_mean, x.shape)
            var = ((x - mean)**2).sum(0).reshape((1, n_feature)) / n
            ret = (x - mean) / ops.broadcast_to(
                ops.power_scalar(var + self.eps, 0.5), x.shape
            ) * ops.broadcast_to(self.weight, x.shape) + ops.broadcast_to(
                self.bias, x.shape)
            return ret
        ### END YOUR SOLUTION


class LayerNorm1d(Module):

    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Tensor(np.ones((dim)), device=device, dtype=dtype)
        self.bias = Tensor(np.zeros((dim)), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = x.shape[0]
        mean = ops.broadcast_to((x.sum(1) / self.dim).reshape((n, 1)), x.shape)
        var = ((x - mean)**2).sum(1).reshape((n, 1)) / self.dim
        ret = (x - mean) / ops.broadcast_to(
            ops.power_scalar(var + self.eps, 0.5), x.shape) * ops.broadcast_to(
                self.weight, x.shape) + ops.broadcast_to(self.bias, x.shape)
        return ret
        ### END YOUR SOLUTION


class Dropout(Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = x.shape
        mask = init.randb(*shape, p=self.p)
        if self.training:
            x = mask * x
            x = x / (1 - self.p)
            return x
        else:
            # idenity function in eval mode
            return x
        ### END YOUR SOLUTION


class Residual(Module):

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
