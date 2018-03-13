from unittest import TestCase
import numpy as np
import dnn_misc
import numpy.testing as test


class TestRelu(TestCase):
    def test_forward(self):
        # check_relu.forward
        np.random.seed(123)
        # example data
        X = np.random.normal(0, 1, (5, 3))
        check_relu = dnn_misc.relu()
        hat_X = check_relu.forward(X)
        ground_hat_X = np.array([[0., 0.99734545, 0.2829785],
                                 [0., 0., 1.65143654],
                                 [0., 0., 1.26593626],
                                 [0., 0., 0.],
                                 [1.49138963, 0., 0.]])

        if (hat_X.shape[0] != 5) or (hat_X.shape[1] != 3):
            print('Wrong output dimension of relu.forward')
        else:
            max_relative_diff = np.amax(np.abs(ground_hat_X - hat_X) / (ground_hat_X + 1e-8))
            print('max_diff_output: ' + str(max_relative_diff))
            if max_relative_diff >= 1e-7:
                print('relu.forward might be wrong')
            else:
                print('relu.forward should be correct')
        print('##########################')

        # check_relu.backward
        grad_hat_X = np.random.normal(0, 1, (5, 3))
        grad_X = check_relu.backward(X, grad_hat_X)
        ground_grad_X = np.array([[-0., 0.92746243, -0.17363568],
                                  [0., 0., -0.87953634],
                                  [0., -0., -1.72766949],
                                  [-0., 0., 0.],
                                  [-0.01183049, 0., 0.]])

        if (grad_X.shape[0] != 5) or (grad_X.shape[1] != 3):
            print('Wrong output dimension of relu.backward')
        else:
            max_relative_diff_X = np.amax(np.abs(ground_grad_X - grad_X) / (ground_grad_X + 1e-8))
            print('max_diff_grad_X: ' + str(max_relative_diff_X))

            if (max_relative_diff_X >= 1e-7):
                print('relu.backward might be wrong')
            else:
                print('relu.backward should be correct')
        print('##########################')


def test_backward(self):
    self.fail()
