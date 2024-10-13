import numpy as np
from functools import wraps


def ensure_2d_tensor(func):
    @wraps(func)
    def wrapper(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return func(self, other)
    return wrapper
class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.atleast_2d(np.array(data, dtype=np.float64))
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    @ensure_2d_tensor
    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    @ensure_2d_tensor
    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * - 1
    
    @ensure_2d_tensor
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other,(int, float)), "power must be a scalar"
        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out 
    
    def __truediv__(self, other):
        return self * (other ** - 1)
    
    @ensure_2d_tensor
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out 
    
    def exp(self):
        x = np.exp(self.data)
        out = Tensor(x, (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out 

    def sum(self):
        x = np.sum(self.data)
        out = Tensor(x, (self,), 'sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out 
    
    def tanh(self):
        x = self.data
        t = (np.exp(2 * x) - 1)/(np.exp(2 * x) + 1)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out 
    
    def sigmoid(self):
        x = self.data
        s = (1 / (1 + np.exp(-x)))
        out = Tensor(s, (self,), 'sigmoid')
        def _backward():
            self.grad += (s * (1 - s))  * out.grad
        out._backward = _backward
        return out
    
    #simplified, not adjusted for numerical stability
    def softmax(self):
        exps = self.exp()
        sum_exps = exps.sum()
        softmax_output = exps / sum_exps
        out = Tensor(softmax_output, (self,), 'softmax')
        return out
