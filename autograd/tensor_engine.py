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
        return f"Tensor(data={self.data}, grad={self.grad}, children={self._prev}, op={self._op})"
    
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
        return self * -1
    
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
        return self * (other ** -1)