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
    b = Tensor([
        [-2],
        [4]
    ])
    c = a + b  
    x = torch.tensor([[0, -2], [3, 4]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[-2], [4]], dtype=torch.float32, requires_grad=True)
    z = x + y  
    result = c.softmax(axis=-1, keepdims=True) 
    s = torch.softmax(z, dim=-1)  
    np.testing.assert_array_almost_equal(result.data, s.detach().numpy(), decimal=5)
    result.backward()  
    grad_output = torch.ones_like(s)  
    s.backward(grad_output)  
    np.testing.assert_array_almost_equal(b.grad, y.grad.numpy(), decimal=5)
    np.testing.assert_array_almost_equal(a.grad, x.grad.numpy(), decimal=5)

def test_cross_entropy_loss():
    a = Tensor([
        [0, -2],
        [3, 4]
    ])
    b = Tensor([
        [-2],
        [4]
    ])
    c = a + b  
    target = Tensor([
        [0, 1],
        [1, 0]
    ]) 
    x = torch.tensor([[0, -2], [3, 4]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[-2], [4]], dtype=torch.float32, requires_grad=True)
    z = x + y 
    target_torch = torch.tensor([1, 0], dtype=torch.int64)  
    result = c.softmax(axis=-1, keepdims=True)
    loss = result.cross_entropy_loss(target, axis=-1)
    loss = loss.mean(axis=0, keepdims=True)
    torch_loss = torch.nn.functional.cross_entropy(z, target_torch)
    np.testing.assert_almost_equal(loss.data, torch_loss.detach().numpy(), decimal=5)
    loss.backward()
    torch_loss.backward()
    np.testing.assert_almost_equal(a.grad, x.grad.numpy(), decimal=5)
    np.testing.assert_almost_equal(b.grad, y.grad.numpy(), decimal=5)