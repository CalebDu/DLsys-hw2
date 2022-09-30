"""Operator implementations."""

import enum
from numbers import Number
from typing import Iterable, Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):

    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return [out_grad]


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):

    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar, )


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION

        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ipt = node.inputs[0]
        grad = out_grad * self.scalar * power_scalar(ipt, self.scalar - 1)
        return [grad]
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -1 * out_grad * a * (b**-2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [out_grad / self.scalar]
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    axis = []

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.axis = list(range(len(a.shape)))
        if self.axes is not None:
            self.axis[self.axes[0]], self.axis[self.axes[1]] = self.axis[
                self.axes[1]], self.axis[self.axes[0]]
        else:
            self.axis[-1], self.axis[-2] = self.axis[-2], self.axis[-1]
        return array_api.transpose(a, self.axis)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ret = transpose(out_grad, axes=self.axes)
        return [ret]
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(a, array_api.ndarray):
            return a.reshape(self.shape)
        else:
            return array_api.array([a]).reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [out_grad.reshape(node.inputs[0].shape)]
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ipt = node.inputs[0]
        grad = out_grad
        for _ in range(len(out_grad.shape) - len(ipt.shape)):
            grad = summation(grad, axes=0)
        for i, dim in enumerate(ipt.shape):
            if dim == 1:
                grad = grad.cached_data.sum(axis=i, keepdims=True)
            grad = Tensor(grad)

        grad = reshape(grad, ipt.shape)
        return [grad]
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ipt = node.inputs[0]
        if self.axes:
            grad = array_api.expand_dims(out_grad.cached_data, self.axes)
            if isinstance(self.axes, int):
                repeat = ipt.shape[self.axes]
                grad = array_api.repeat(grad, repeat, self.axes)
            else:
                repeat = []
                for i in self.axes:
                    repeat.append(ipt.shape[i])
                for r, a in zip(repeat, self.axes):
                    grad = array_api.repeat(grad, r, a)
        else:
            grad = array_api.ones_like(ipt.cached_data) * out_grad.cached_data

        grad = Tensor(grad)

        return [grad]
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        da = matmul(out_grad, transpose(b))
        db = matmul(transpose(a), out_grad)
        if len(da.shape) is not len(a.shape):
            da = summation(da, tuple(range(len(da.shape) - len(a.shape))))
        if len(db.shape) is not len(b.shape):
            db = summation(db, tuple(range(len(db.shape) - len(b.shape))))
        assert (da.shape == a.shape)
        assert (db.shape == b.shape)
        return [da, db]
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.multiply(a, -1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [out_grad * -1]
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ipt = node.inputs[0]
        return [out_grad / ipt]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ipt = node.inputs[0]
        # t = type(Tensor)
        # t = type(exp(ipt))
        grad = out_grad * exp(ipt)
        # t = type(grad)
        return [grad]
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        mask = (a > 0)
        ret = a * mask
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ipt = node.inputs[0]
        grad = Tensor(
            (ipt.cached_data > 0).astype(array_api.float32)) * out_grad
        return [grad]
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
