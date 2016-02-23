from microscopium._util import generate_spiral
import pytest
import numpy as np

spiral_3_down_anticlockwise = [[6, 5, 4],
                               [7, 0, 3],
                               [8, 1, 2]]

spiral_5_right_clockwise = [[20, 21, 22, 23, 24],
                            [19,  6,  7,  8,  9],
                            [18,  5,  0,  1, 10],
                            [17,  4,  3,  2, 11],
                            [16, 15, 14, 13, 12]]

spiral_5_up_anticlockwise = [[12, 11, 10,  9, 24],
                             [13,  2,  1,  8, 23],
                             [14,  3,  0,  7, 22],
                             [15,  4,  5,  6, 21],
                             [16, 17, 18, 19, 20]]

spiral_7_left_clockwise = [[30, 31, 32, 33, 34, 35, 36],
                           [29, 12, 13, 14, 15, 16, 37],
                           [28, 11,  2,  3,  4, 17, 38],
                           [27, 10,  1,  0,  5, 18, 39],
                           [26,  9,  8,  7,  6, 19, 40],
                           [25, 24, 23, 22, 21, 20, 41],
                           [48, 47, 46, 45, 44, 43, 42]]

spiral_3_down_anticlockwise_params = ("spiral_3_left_clockwise", {
    "dim": 3,
    "direction": "down",
    "clockwise": False,
    "expected": spiral_3_down_anticlockwise
})

spiral_5_right_clockwise_params = ("spiral_3_left_clockwise", {
    "dim": 5,
    "direction": "right",
    "clockwise": True,
    "expected": spiral_5_right_clockwise
})

spiral_5_up_anticlockwise_params = ("spiral_3_left_clockwise", {
    "dim": 5,
    "direction": "up",
    "clockwise": False,
    "expected": spiral_5_up_anticlockwise
})

spiral_7_left_clockwise_params = ("spiral_7_left_clockwise", {
    "dim": 7,
    "direction": "left",
    "clockwise": True,
    "expected": spiral_7_left_clockwise
})


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        argnames = list(scenario[1].keys())
        argvalues.append((list(scenario[1].values())))
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


class TestSeries:
    scenarios = [spiral_3_down_anticlockwise_params,
                 spiral_5_right_clockwise_params,
                 spiral_5_up_anticlockwise_params,
                 spiral_7_left_clockwise_params]

    def test_generate_spiral(self, dim, direction, clockwise,
                                    expected):
        actual = generate_spiral(dim, direction, clockwise)
        np.testing.assert_array_equal(actual, expected)

