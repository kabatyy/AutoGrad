import numpy as np
from autograd.core.engine import Tensor
import pytest

def test_addition():
    a = Tensor([
        [1,2], 
        [3,4]])
    b = Tensor([
        [5,6],
        [7,8]])
    result = a + b
    np.testing.assert_array_equal(result.data, np.array([[6, 8], [10, 12]]) )

    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_equal(a.grad, np.ones_like(a.data))
    np.testing.assert_array_equal(b.grad, np.ones_like(b.data))

def test_multiplication():
    a = Tensor([
        [1,2],
        [3,4]])
    b = Tensor([
        [5,6],
        [7,8]])
    result = a * b
    np.testing.assert_array_equal(result.data, np.array([[5, 12], [21, 32]]))

    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_equal(a.grad, b.data)
    np.testing.assert_array_equal(b.grad, a.data)

def test_matrix_multiplication():
    a = Tensor([
        [1,2],
        [3,4]])
    b = Tensor([
        [5,6],
        [7,8]])
    result = a @ b
    np.testing.assert_array_equal(result.data, np.array([[19, 22], [43, 50]]))
    
    result.grad = np.ones_like(result.data)
    result._backward()
    np.testing.assert_array_equal(a.grad, np.array([[11, 15], [11, 15]]))
    np.testing.assert_array_equal(b.grad, np.array([[4, 4], [6, 6]]))

    



