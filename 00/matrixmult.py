#! /usr/bin/env python3

# how to run:
#    pytest-3 matrixmult.py


import pytest
import numpy as np

def multiply(a, b):
    if len(a[0])!=len(b):
        raise Exception("Wrong matrix multiplication")
    result = []
    for i in range(len(a)):
        mid_result = []
        for j in range(len(b[0])):
            single_field = 0
            for k in range(len(b)):
                single_field += a[i][k] * b[k][j]
            mid_result.append(single_field)
        result.append(mid_result)
    return result


def multiplyWithNumpy(a, b):
    return np.dot(a, b)


testdata = [
    ([[3]], [[6]], [[18]]),
    ([[1, 2]],   [[3], [4]], [[11]]),
    ([[1, 2, 3], [4, 5, -7]],  [[3, 0, 4], [2, 5, 1], [-1, -1, 0]],  [[4, 7, 6], [29, 32, 21]])
]
@pytest.mark.parametrize("multiplicationFunc", [multiply, multiplyWithNumpy])
@pytest.mark.parametrize("a,b,expected", testdata)
def test_multiply(multiplicationFunc, a, b, expected):
    assert np.array_equal(multiplicationFunc(a, b), expected)


testdataImpossibleProducts = [
    ([[3, 4]], [[6, 7]]),
    ([[1, 2]], [[3, 4], [4, 5], [5, 6]])
]
@pytest.mark.parametrize("multiplicationFunc", [multiply, multiplyWithNumpy])
@pytest.mark.parametrize("a,b", testdataImpossibleProducts)
def test_multiplyIncompatibleMatrices(multiplicationFunc, a, b):
    with pytest.raises(Exception) as e:
        multiplicationFunc(a, b)



