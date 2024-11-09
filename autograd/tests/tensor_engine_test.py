import numpy as np
from autograd.core.engine import Tensor
import torch
import pytest

def test_addition():
    a = Tensor([
        [1, 2], 
        [3, 4]])
    b = Tensor([
        [5, 6],
        [7, 8]])
    result = a + b
    np.testing.assert_array_equal(result.data, np.array([[6, 8], [10, 12]]) )

    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
    np.testing.assert_array_equal(b.grad, np.ones_like(b.data))

def test_multiplication():
    a = Tensor([
        [1, 2],
        [3, 4]])
    b = Tensor([
        [5, 6],
        [7, 8]])
    result = a * b
    np.testing.assert_array_equal(result.data, np.array([[5, 12], [21, 32]]))

    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_equal(a.grad, b.data)
    np.testing.assert_array_equal(b.grad, a.data)

def test_matrix_multiplication():
    a = Tensor([
        [1, 2],
        [3, 4]])
    b = Tensor([
        [5, 6],
        [7, 8]])
    result = a @ b
    np.testing.assert_array_equal(result.data, np.array([[19, 22], [43, 50]]))
    
    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_equal(a.grad, np.array([[11, 15], [11, 15]]))
    np.testing.assert_array_equal(b.grad, np.array([[4, 4], [6, 6]]))

def test_relu():
    a = Tensor([
        [-1, 2],
        [0, -3]
    ])
    result = a.relu()
    np.testing.assert_array_equal(result.data, np.array([[0,2],[0,0]]))

    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_equal(a.grad, np.array([[0,1],[0,0]]))

def test_exp():
    a = Tensor([
        [1, 2],
        [3, 4]])
    result = a.exp()
    expected = np.exp(a.data)
    np.testing.assert_array_equal(result.data, expected)

    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_equal(a.grad, result.data)

def test_tanh():
    a = Tensor([
        [0, -2],
        [3, 4]
    ])
    result = a.tanh()
    t = np.tanh(a.data)
    np.testing.assert_array_almost_equal(result.data, t)

    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_almost_equal(a.grad, (1 - t**2) * result.grad)

def test_sigmoid():
     a = Tensor([
        [0, -2],
        [3, 4]
    ])
     result = a.sigmoid()
     x = torch.Tensor([[0, -2],[3, 4]])
     s = torch.sigmoid(x)
     np.testing.assert_array_almost_equal(result.data, s)

     result.grad = np.ones_like(result.data)
     result._backward()
     np.testing.assert_array_almost_equal(a.grad, s * (1 -s) * result.grad)

def test_softmax():
      a = Tensor([
        [0, -2],
        [3, 4]
    ])
      result = a.softmax(axis=-1, keepdims=True)
      x = torch.Tensor([[0, -2],[3, 4]])
      s = torch.softmax(x, dim=-1)
      np.testing.assert_array_almost_equal(result.data, s)
    
      
