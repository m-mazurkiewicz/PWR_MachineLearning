#! /usr/bin/env python3

# how to run:
#    pytest-3 matrixmult.py


import pytest
import numpy as np


def multiply(a, b):
    if len(a[0])!=len(b):
        raise Exception
    new_matrix = []
    for row_index_a, row_a in enumerate(a):
        new_row = []
        for column_index_b in range(len(b[0])):
            new_element = 0
            for column_index_a, element_a in enumerate(row_a):
                new_element += element_a*b[column_index_a][column_index_b]
            new_row.append(new_element)
        new_matrix.append(new_row)
    return new_matrix


def multiply_with_numpy(a, b):
    return np.dot(a,b)


test_data = [
    ([[3]], [[6]], [[18]]),
    ([[1, 2]],   [[3], [4]], [[11]]),
    ([[1, 2, 3], [4, 5, -7]],  [[3, 0, 4], [2, 5, 1], [-1, -1, 0]],  [[4, 7, 6], [29, 32, 21]])
]


@pytest.mark.parametrize("multiplicationFunc", [multiply, multiply_with_numpy])
@pytest.mark.parametrize("a,b,expected", test_data)
def test_multiply(multiplicationFunc, a, b, expected):
    assert np.array_equal(multiplicationFunc(a, b), expected)


test_data_impossible_products = [
    ([[3, 4]], [[6, 7]]),
    ([[1, 2]], [[3, 4], [4, 5], [5, 6]])
]


@pytest.mark.parametrize("multiplication_func", [multiply, multiply_with_numpy])
@pytest.mark.parametrize("a,b", test_data_impossible_products)
def test_multiply_incompatible_matrices(multiplication_func, a, b):
    with pytest.raises(Exception) as e:
        multiplication_func(a, b)

