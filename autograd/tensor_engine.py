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
    
    
        