import math
import numpy as np
from functools import wraps

class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self._prev=set(_children)
        self._op=_op
        self.label=label
        self.grad=0.0
        self._backward=lambda:None
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out 
    
    def __radd__(self,other):
        return self+other
    
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
    
    def __rmul__(self,other):
        return self*other
    
    def __neg__(self):
        return self *-1
    
    def __sub__(self,other):
        out=self+(-other)
        return out 
    def __pow__(self,other):
        assert isinstance(other,(int,float)),'Value is not supported as a power, use a float or an int'
        out=Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad+=other * (self.data ** (other-1)) * out.grad
        out._backward=_backward
        return out 
    def __truediv__(self,other):
        return self * other**-1
    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(t,(self,),'tanh')
        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        return out 
    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self,),'exp')
        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()


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
        self.shape = np.shape(self)
    
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

    def sum(self, axis=None, keepdims=False):
        x = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(x, (self,), 'sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out 
    
    def max(self, axis=None, keepdims=False):
        max_vals = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(max_vals, (self,), 'max')
        def _backward():
            grad_mask = (self.data == max_vals)
            num_maxes = grad_mask.sum(axis=axis, keepdims=keepdims)
            grad_mask = grad_mask.astype(np.float64) / num_maxes
            self.grad += grad_mask * out.grad
        out._backward = _backward
        return out
   
    def log(self):
            x= self.data + 1e-12
            log = np.log(x)
            out = Tensor(log, (self,), 'log')
            def _backward():
                self.grad += (1 / x) * out.grad
            out._backward =_backward
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
        softmax_output = (exps.data / sum_exps.data)
        out = Tensor(softmax_output, (self,), 'softmax')
        return out
   
    @ensure_2d_tensor
    def cross_entropy_loss(self, target):
        log_probs = self.log()
        loss = -((target.data * log_probs.data).sum())
        out = Tensor(loss, (self, target), 'cross-entropy loss')
        return out 
