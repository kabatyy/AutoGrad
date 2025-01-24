import math
import numpy as np
from functools import wraps
from  .utils import broadcast_axis
from typing import *
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


class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.atleast_2d(np.array(data, dtype=np.float64))
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.shape = np.shape(self.data)
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def broadcast_to(self, shape: Tuple[int]):
        new_shape = np.broadcast_to(self.data, shape)
        out = Tensor(new_shape, (self,), f'broadcast_to {shape}')
        broadcasted_axes = broadcast_axis(self.shape, shape)[0]
        
        def _backward():
            self.grad += np.sum(out.grad, axis=broadcasted_axes, keepdims=True)
        out._backward = _backward
        return out
   
    def _preprocess_binop(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.shape == other.shape:
            return self, other
        else:
            broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
            self, other = self.broadcast_to(broadcast_shape), other.broadcast_to(broadcast_shape)
            return self, other
        
    def __add__(self, other):
        self, other = self._preprocess_binop(other)
        out = Tensor(self.data + other.data, (self, other), '+')
       
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    

    def __mul__(self, other):
        self, other = self._preprocess_binop(other)
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
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
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
    
    def mean(self, axis=None, keepdims=True):
        if axis:
            n = self.data.shape[axis]
        else:
            n = self.data.size 
        out = (self.sum(axis=axis, keepdims=keepdims)) / n
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
      
    def softmax(self, axis=None, keepdims=True):
        max_val = self.max(axis=axis, keepdims=keepdims)
        exps = (self - max_val).exp()
        sum_exps = exps.sum(axis=axis, keepdims=keepdims)
        out = exps / sum_exps
        return out
   
    def cross_entropy_loss(self, target, axis=None, keepdims=True):
        log_probs = self.log()
        out = -((target * log_probs).sum(axis=axis, keepdims=keepdims))
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
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()