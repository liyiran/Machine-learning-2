from unittest import TestCase
from logistic import binary_train, \
    binary_predict, generate_y, OVR_train, OVR_predict, multinomial_predict, multinomial_train
import numpy as np
import numpy.testing as test


class TestGradient_descent(TestCase):
    def test_dot_product(self):
        x = np.matrix('1; 2; 3')
        y = np.matrix('4 5 6; 7 8 9; 1 2 3')
        # print(np.multiply(2, y))
        # print(np.subtract(y, np.max(y)))
        # print(np.sum(np.multiply(x,y),axis=1))
        # print()
        # print(np.append(np.ones((y.shape[0], 1)), y, axis= 1))
        x = np.matrix(np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]))
        y = np.matrix(np.array([[0, 1, 0], [1, 0, 1]]))
        w = np.matrix(np.array([[10, 11, 12], [20, 21, 22]]))
        y_input = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
        # print(x * y)
        weight = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # print(weight)
        # print(weight.reshape(3,1,3) * y)
        # test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(np.einsum("CD,ND->CN", w, x) - y)
        # print(x * np.tile(y_input, (3, 1, 1)))

    def test_set_y(self):
        y = np.array([1, 2, 3])
        print(y.shape)
        print(generate_y(y, 3, 5).shape)
        big_y = generate_y(y, 3, 5)
        print(big_y)
        # x = 0
        # new_y = np.empty((0))
        # for r in big_y:
        #     # print((r == x).astype(int))
        #     new_y = np.append(new_y, (r == x).astype(int))
        #     x += 1
        # print(new_y.reshape(big_y.shape))

    def test_binary_classifier_integration(self):
        """
          Inputs:
          - X: training features, a N-by-D numpy array, where N is the 
          number of training points and D is the dimensionality of features
          - y: binary training labels, a N dimensional numpy array where 
          N is the number of training points, indicating the labels of 
          training data
          - step_size: step size (learning rate)
        
          Returns:
          - w: D-dimensional vector, a numpy array which is the weight 
          vector of logistic regression
          - b: scalar, which is the bias of logistic regression
        
          Find the optimal parameters w and b for inputs X and y.
          Use the average of the gradients for all training examples to
          update parameters.
          """
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [7], [8], [9]])
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        w, b = binary_train(X, y, max_iterations=5000)
        # self.assertAlmostEqual(12.16800695, b)
        # test.assert_array_almost_equal([[-22.3208943]], w)
        pred = binary_predict(X, w, b)
        test.assert_array_equal(y, pred)
        test_x = np.array([[9], [8], [7], [6], [5], [4], [3], [2], [1], [10]])
        test_y = binary_predict(test_x, w, b)
        test.assert_array_equal(np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]), test_y)

    def test_ovr_integration(self):
        X = np.array([[0.1], [0.2], [0.3]])
        y = np.array([0, 1, 2])
        w, b = OVR_train(X, y, 3)
        # print(w, b)
        test.assert_array_equal(y, OVR_predict(X, w, b))
        # w, b = multinomial_train_old(X, y, 3)
        # print(w,b)
        w, b = multinomial_train(X, y, 3)
        print(w, b)
        test.assert_array_equal(y, multinomial_predict(X, w, b))

    def accuracy_score(self, train_lable, preds):
        return np.sum(train_lable == preds).astype(float) / len(train_lable)

    def test_ovr_integration_big(self):
        from data_loader import toy_data_multiclass_3_classes_non_separable, \
            toy_data_multiclass_5_classes, \
            data_loader_mnist

        datasets = [(toy_data_multiclass_3_classes_non_separable(),
                     'Synthetic data', 3),
                    (toy_data_multiclass_5_classes(), 'Synthetic data', 5)
            ,
                    (data_loader_mnist(), 'MNIST', 10)
                    ]

        for data, name, num_classes in datasets:
            print('%s: %d class classification' % (name, num_classes))
            X_train, X_test, y_train, y_test = data

            # print('One-versus-rest:')
            # w, b = OVR_train(X_train, y_train, C=num_classes)
            # train_preds = OVR_predict(X_train, w=w, b=b)
            # preds = OVR_predict(X_test, w=w, b=b)
            # print('train acc: %f, test acc: %f' %
            #       (self.accuracy_score(y_train, train_preds),
            #        self.accuracy_score(y_test, preds)))

            print('Multinomial:')
            w, b = multinomial_train(X_train, y_train, C=num_classes)
            train_preds = multinomial_predict(X_train, w=w, b=b)
            preds = multinomial_predict(X_test, w=w, b=b)
            print('train acc: %f, test acc: %f' %
                  (self.accuracy_score(y_train, train_preds),
                   self.accuracy_score(y_test, preds)))

    def test_binary_integration(self):
        from data_loader import toy_data_binary, \
            data_loader_mnist

        print('Performing binary classification on synthetic data')
        X_train, X_test, y_train, y_test = toy_data_binary()

        w, b = binary_train(X_train, y_train)

        train_preds = binary_predict(X_train, w, b)
        preds = binary_predict(X_test, w, b)
        print('train acc: %f, test acc: %f' %
              (self.accuracy_score(y_train, train_preds),
               self.accuracy_score(y_test, preds)))

        print('Performing binary classification on binarized MNIST')
        X_train, X_test, y_train, y_test = data_loader_mnist()

        binarized_y_train = [0 if yi < 5 else 1 for yi in y_train]
        binarized_y_test = [0 if yi < 5 else 1 for yi in y_test]

        w, b = binary_train(X_train, binarized_y_train)

        train_preds = binary_predict(X_train, w, b)
        preds = binary_predict(X_test, w, b)
        print('train acc: %f, test acc: %f' %
              (self.accuracy_score(binarized_y_train, train_preds),
               self.accuracy_score(binarized_y_test, preds)))
