"""Optimization module"""
from math import sqrt
import needle as ndl
import numpy as np
import torch.optim.adam


class Optimizer:

    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for para in self.params:
            u = self.u.get(para)
            grad = para.grad.detach() + self.weight_decay * para.detach()
            u_new = grad * (1 - self.momentum) + self.momentum * (
                0 if u == None else u)
            id = para.detach()
            self.u[para] = u_new
            d_para = u_new 
            para.cached_data -= (self.lr * d_para).cached_data
        ### END YOUR SOLUTION


class Adam(Optimizer):

    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for para in self.params:
            m = self.m.get(para)
            v = self.v.get(para)
            grad = para.grad.detach() + self.weight_decay * para.detach()
            m_new = (1 -
                     self.beta1) * grad + self.beta1 * (0 if m == None else m)
            v_new = (1 - self.beta2) * (grad**2) + self.beta2 * (0 if v == None
                                                                 else v)
            self.m[para] = m_new
            self.v[para] = v_new

            m_new = (m_new / (1 - (self.beta1**self.t))).detach()
            v_new = (v_new / (1 - (self.beta2**self.t))).detach()
            para.cached_data -= self.lr * ((m_new / (
                (v_new**0.5) + self.eps))).cached_data

        ### END YOUR SOLUTION
