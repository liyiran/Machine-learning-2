from unittest import TestCase
import dnn_misc
import numpy as np
import numpy.testing as test


class TestLinear_layer(TestCase):

    def test_linear(self):
        np.random.seed(123)
        # example data
        X = np.random.normal(0, 1, (5, 3))

        # example modules
        check_linear = dnn_misc.linear_layer(input_D=3, output_D=2)
        check_relu = dnn_misc.relu()
        check_dropout = dnn_misc.dropout(r=0.5)

        # check_linear.forward
        hat_X = check_linear.forward(X)
        ground_hat_X = np.array([[0.42525407, -0.2120611],
                                 [0.15174804, -0.36218431],
                                 [0.20957104, -0.57861084],
                                 [0.03460477, -0.35992763],
                                 [-0.07256568, 0.1385197]])

        if (hat_X.shape[0] != 5) or (hat_X.shape[1] != 2):
            print('Wrong output dimension of linear.forward')
            self.fail()
        else:
            max_relative_diff = np.amax(np.abs(ground_hat_X - hat_X) / (ground_hat_X + 1e-8))
            print('max_diff_output: ' + str(max_relative_diff))
            if max_relative_diff >= 1e-7:
                print('linear.forward might be wrong')
                self.fail()
            else:
                print('linear.forward should be correct')
                pass
        # check_linear.backward
        grad_hat_X = np.random.normal(0, 1, (5, 2))
        grad_X = check_linear.backward(X, grad_hat_X)

        ground_grad_X = np.array([[-0.32766959, 0.13123228, -0.0470483],
                                  [0.22780188, -0.04838436, 0.04225799],
                                  [0.03115675, -0.32648556, -0.06550193],
                                  [-0.01895741, -0.21411292, -0.05212837],
                                  [-0.26923074, -0.78986304, -0.23870499]])

        ground_grad_W = np.array([[-0.27579345, -2.08570514],
                                  [4.52754775, -0.40995374],
                                  [-1.2049515, 1.77662551]])

        ground_grad_b = np.array([[-4.55094716, -2.51399667]])

        if (grad_X.shape[0] != 5) or (grad_X.shape[1] != 3):
            print('Wrong output dimension of linear.backward')
            self.fail()
        else:
            max_relative_diff_X = np.amax(np.abs(ground_grad_X - grad_X) / (ground_grad_X + 1e-8))
            print('max_diff_grad_X: ' + str(max_relative_diff_X))
            max_relative_diff_W = np.amax(np.abs(ground_grad_W - check_linear.gradient['W']) / (ground_grad_W + 1e-8))
            print('max_diff_grad_W: ' + str(max_relative_diff_W))
            max_relative_diff_b = np.amax(np.abs(ground_grad_b - check_linear.gradient['b']) / (ground_grad_b + 1e-8))
            print('max_diff_grad_b: ' + str(max_relative_diff_b))

            if (max_relative_diff_X >= 1e-7) or (max_relative_diff_W >= 1e-7) or (max_relative_diff_b >= 1e-7):
                print('linear.backward might be wrong')
                self.fail()
            else:
                print('linear.backward should be correct')
                pass

    def test_eisum(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([[7, 8], [9, 10], [11, 12]])
        print(np.einsum('ji,ja->ia', x, y))
