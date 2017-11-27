import os
import sys
import h5py
import pickle
import traceback
import numpy as np
from scipy import misc, ndimage


def load_dataset():
    train_dataset = h5py.File('cat_not_cat_dataset/train_catvsnotcat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

    test_dataset = h5py.File('cat_not_cat_dataset/test_catvsnotcat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

    classes = np.array(test_dataset["list_classes"][:])  # list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid_fn(z):
    """
    Calculates the sigmoid value for the matrix / vector passed as argument
    :param z: the matrix / vector for which sigmoid needs to be calculated
    :return: sigmoid value
    """

    return 1 / (1 + np.exp(- z))


def initialize(dimension):
    """
    initialize weight and b
    :param dimension: dimension for weight (same as number of features in a sample)
    :return: w and b
    """

    w = np.zeros((dimension, 1))
    b = 0.0

    return w, b


def calculate_cost_and_gradients(w, b, X, Y):
    """
    Calculate cost and  gradient for w, b considering cost function
    :param w: weight matrix
    :param b: bias
    :param X: feature matrix of shape (dimension, sample_size)
    :param Y: vector with true value for the labels ( 0 or 1 )
    :return: cost - negative log-likelihood cost for logistic regression
             dict:  w_gradient - gradient of the loss with respect to w
                    b_gradient - gradient of the loss with respect to b
    """

    # m : number of samples
    m = X.shape[1]

    # compute activation function
    # activation uses w and b , hence new activation every time for new values of w and b
    predicted = sigmoid_fn(np.dot(w.T, X) + b)

    # cost - negative log-likelihood cost for logistic regression
    cost = 1 / m * np.sum(-Y * np.log(predicted) - (1 - Y) * np.log(1 - predicted))

    # w_gradient - gradient of the loss with respect to w
    # matrix multiplication does take care of the summation for each weight
    # for each sample -> gradient for each weight corresponding to respective features of each sample
    # d cost/ w_gradient1 = 1/m* sum( (predicted(i) - true value(i)) * x1(i) )
    # d cost/ w_gradient2 = 1/m* sum( (predicted(i) - true value(i)) * x2(i) )

    w_gradient = 1 / m * np.dot(X, (predicted - Y).T)

    # b_gradient - gradient of the loss with respect to b
    # calculating for each sample
    # d cost/ b_gradient = 1/m * sum( (predicted(i) - true value(i)) * 1)

    b_gradient = 1 / m * np.sum((predicted - Y))

    # assertion will report error if shapes are mismatch
    assert (w_gradient.shape == w.shape)
    cost = np.squeeze(cost)  # squeeze removes single dimensional entries eg: (1, 3, 1) gets converted to (3,)
    assert (cost.shape == ())

    gradients = {"w_gradient": w_gradient,
                 "b_gradient": b_gradient}

    return gradients, cost


# gradient descent to optimize params
def optimize_params(w, b, X, Y, iterations, learning_rate):
    costs = []
    w_gradient, b_gradient = None, None

    for i in range(iterations):
        gradients, cost = calculate_cost_and_gradients(w, b, X, Y)

        w_gradient = gradients['w_gradient']
        b_gradient = gradients['b_gradient']

        # update rules
        w = w - learning_rate * w_gradient
        b = b - learning_rate * b_gradient

        if i % 100 == 0:
            costs.append(cost)
            print(' \n Cost after %s iterations = %s' % (i, cost))
    # for loop ends

    params = {'w': w,
              'b': b}

    gradients = {"w_gradient": w_gradient,
                 "b_gradient": b_gradient}

    return params, gradients, costs


def predict(w, b, X_new):
    """
    Prediction for new data
    :param w: optimized weight
    :param b: optimized b
    :param X_new: new samples
    :param Y_new_true: true value for new samples
    :return: predicted result after checking of threshold value
    """

    m = X_new.shape[1]
    result = np.zeros((1, m))

    predicted_new = sigmoid_fn(np.dot(w.T, X_new) + b)

    for i in range(result.shape[1]):
        result[0][i] = 0 if predicted_new[0][i] <= 0.5 else 1

    return result


def final_model(X_train, Y_train, iterations, learning_rate):
    """
    Combines all steps for the NN model
    :param X_train: train data ste
    :param Y_train: true values for train data set
    :param X_test: test data set
    :param Y_test: true values for test data set
    :param iterations: number of iterations for gradient descent optimizer
    :param learning_rate: the learning rate of the model
    :return: None
    """

    dimension = X_train.shape[0]
    w, b = initialize(dimension=dimension)

    parameters, gradients, costs = optimize_params(w=w, b=b, X=X_train, Y=Y_train, iterations=iterations,
                                                   learning_rate=learning_rate)

    w_optimized = parameters['w']
    b_optimized = parameters['b']

    param_dict = {'dimension': dimension, 'w': w_optimized, 'b': b_optimized}

    with open('params.pkl', 'wb') as f:
        pickle.dump(param_dict, f)

    prediction_train = predict(w=w_optimized, b=b_optimized, X_new=X_train)

    print('\n Accuracy for train set : ', np.mean(prediction_train == Y_train) * 100)


if __name__ == '__main__':
    if sys.argv[1].lower() == 'train':
        train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()

        # reshaping train and test data sets from (sample_numbers, height, width, 3) to (height*width*3, sample_numbers)
        #  3 dimension corresponds to RGB color channel values
        train_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T

        # normalize features.. by max value ( 255 for each pixel)
        train_x = train_x_flatten / 255

        final_model(X_train=train_x, Y_train=train_set_y, iterations=10000, learning_rate=0.001)

    elif sys.argv[1].lower() == 'test':
        train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()

        # reshaping train and test data sets from (sample_numbers, height, width, 3) to (height*width*3, sample_numbers)
        #  3 dimension corresponds to RGB color channel values
        test_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T

        # normalize features.. by max value ( 255 for each pixel)
        test_x = test_x_flatten / 255

        if not os.path.isfile('params.pkl'):
            sys.exit('\n The model has not been trained before. No params pickle file found \n To train run: $python main.py train')

        with open('params.pkl', 'rb') as f:
            params_dict = pickle.load(f)

        prediction_test = predict(w=params_dict['w'], b=params_dict['b'], X_new=test_x)

        print('\n Accuracy for test set : ', np.mean(prediction_test == test_set_y) * 100)

    elif sys.argv[1].lower() == 'classify':
        if len(sys.argv) != 3:
            sys.exit('\n Wrong number of arguments for predict.'
                     ' \n "classify" option requires image path for classification \n')

        file = sys.argv[2]

        if not os.path.isfile(file):
            sys.exit('\n Image file does not exist.')

        if not os.path.isfile('params.pkl'):
            sys.exit('\n params.pkl is missing')

        with open('params.pkl', 'rb') as f:
            params_dict = pickle.load(f)

        dimension = params_dict['dimension']
        w = params_dict['w']
        b = params_dict['b']
        num_px = 64

        try:
            image = np.array(ndimage.imread(file, flatten=False), dtype=np.float64)
            my_image = misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
        except Exception:
            sys.exit(traceback.format_exc())
        else:
            prediction = predict(w=params_dict['w'], b=params_dict['b'], X_new=my_image)
            if prediction[0] == 1:
                print('\n Its a cat image')
            else:
                print('\n Its NOT a cat image')    
    else:
        sys.exit('\n Maybe wrong arguments or wrong number of arguments. \n Possible values :\n 1:train \n 2:test \n 3:classify (requires another argument for the image file)')
