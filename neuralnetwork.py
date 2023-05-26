import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score


# fully - connected layer
class FCLayer:
    def __init__(self, input_size, output_size, seed=None):
        self.input = None
        self.output = None
        if seed is not None:
            np.random.seed(seed)
        # uniform distribution [0, 1] - 0.5 -> [-0.5, 0.5]
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # forward propagation:
    # compute output Y of the layer for a given input X
    def forward_propagation(self, input):
        if (np.ndim(input) == 1):
            self.input = input.reshape((1, input.shape[0]))
        else:
            self.input = input

        # Y = XW + B
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # backward propagation:
    # compute dE/dW, dE/dB (update weights+bias) for a given output_error=dE/dY
    # return input_error=dE/dX
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error

    # this method will be used in case we train a regularized model
    def backward_propagation_ridge(self, output_error, learning_rate, reg_lambda):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # regularization with the gradient of the ridge formula
        weights_error += 2 * reg_lambda * self.weights

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error


# activation layer
class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.input = None
        self.output = None
        self.activation = activation
        self.activation_prime = activation_prime

    # forward propagation:
    # return the activated input (activation fcn applied on input)
    def forward_propagation(self, input_data):
        if (np.ndim(input_data) == 1):
            self.input = input_data.reshape((1, input_data.shape[0]))
        else:
            self.input = input_data

        self.output = self.activation(self.input)
        return self.output

    # backward propagation:
    # return input_error=dE/dX for a given output_error=dE/dY
    # learning_rate not used, there are no parameters to update
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to the network
    def add(self, layer):
        self.layers.append(layer)

    # set loss function
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        samples = len(input_data)  # input size
        result = []

        # run network for all samples
        for i in range(0, samples):
            # forward propagation
            output = input_data[i, :]  # i-th sample
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)  # length of the training set
        err_vect = np.zeros(epochs)
        # training loop
        for i in range(0, epochs):
            err = 0
            for j in range(0, samples):  # through all training samples
                # forward propagation
                output = x_train[j, :]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # average error per sample
            err /= samples
            err_vect[i] = err
            # err_vect = np.append(err_vect, err)  # append error to the array
            # print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        return err_vect

    # method used to fit the model with ridge
    def fit_plus_ridge(self, x_train, y_train, epochs, learning_rate, reg_lambda):
        samples = len(x_train)  # length of the training set
        err_vect = np.zeros(epochs)
        # training loop
        for i in range(0, epochs):
            err = 0
            for j in range(0, samples):  # through all training samples
                # forward propagation
                output = x_train[j, :]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    # in case the layer is a hidden layer, we apply backwardpropagation using ridge

                    if isinstance(layer, FCLayer):
                        error = layer.backward_propagation_ridge(
                            error, learning_rate, reg_lambda)
                    else:
                        error = layer.backward_propagation(
                            error, learning_rate)

            # average error per sample
            err /= samples
            err_vect[i] = err
            # print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        return err_vect

    def fit_plus_validation(self, x_train, y_train, x_val, y_val, epochs, learning_rate):
        samples = len(x_train)  # length of the training set
        err_vect_train = np.zeros(epochs)
        err_vect_val = np.zeros(epochs)
        # training loop
        for i in range(0, epochs):
            err = 0
            for j in range(0, samples):  # through all training samples
                # forward propagation
                output = x_train[j, :]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # average error per sample (at the end of training epoch)

            err = 0
            for j in range(0, samples):
                # forward propagation
                output = x_train[j, :]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                # compute loss
                err += self.loss(y_train[j], output)
            err /= samples
            err_vect_train[i] = err

            err = 0
            for j in range(0, x_val.shape[0]):
                # forward propagation
                output = x_val[j, :]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                # compute loss
                err += self.loss(y_val[j], output)
            err /= x_val.shape[0]
            err_vect_val[i] = err

        return err_vect_train, err_vect_val

    def fit_batch(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)  # length of the training set
        err_vect = np.zeros(epochs)
        # training loop
        for i in range(0, epochs):
            err = 0
            error_vect = np.zeros([samples, y_train.shape[1]])
            for j in range(0, samples):  # through all training samples
                # forward propagation
                output = x_train[j, :]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)
                error_vect[j, :] = self.loss_prime(y_train[j], output)

            # backward propagation - after all input samples have been propagated forwardly
            error = error_vect.mean(axis=0)

            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, learning_rate)

            # average error per sample (at the end of training epoch)
            err /= samples
            err_vect[i] = err
        return err_vect

    def fit_mini_batch(self, x_train, y_train, epochs, learning_rate, batch_size):
        samples = len(x_train)  # length of the training set
        err_vect = np.zeros(epochs)
        # training loop
        for i in range(0, epochs):
            err = 0
            batch_i = 0
            output_batch = np.zeros([batch_size, y_train.shape[1]])
            error_vect = np.zeros([batch_size, y_train.shape[1]])

            for j in range(0, samples):  # through all training samples
                # forward propagation
                output = x_train[j, :]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                output_batch[batch_i, :] = output
                error_vect[batch_i, :] = self.loss_prime(y_train[j, :], output)

                # compute loss
                err += self.loss(y_train[j], output)

                if (j % batch_size == batch_size - 1):  # -> update weights
                    # backward propagation - after batch_size samples have been propagated forwardly
                    error = error_vect.mean(axis=0)
                    
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(error, learning_rate)
                    batch_i = 0
                    output_batch = np.zeros([batch_size, y_train.shape[1]])
                    error_vect = np.zeros([batch_size, y_train.shape[1]])
                else:
                    batch_i += 1  # continue

            # average error per sample (at the end of training epoch)

            err = 0
            for j in range(0, samples):
                # forward propagation
                output = x_train[j, :]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                # compute loss
                err += self.loss(y_train[j], output)
            err /= samples
            err_vect[i] = err

        return err_vect

    def plot_error_curve(self, err_vect):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(err_vect)+1, 1),
                 err_vect, '-', color='darkblue')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.title('Average error per sample through training epochs')

    def nn_evaluate_binary(self, x_train, y_train, x_test, y_test, epochs, learning_rate, mode=None, batch_size=None):
        if mode == 'batch':
            self.fit_batch(x_train, y_train, epochs, learning_rate)
        elif mode == 'ridge':
            self.fit_plus_ridge(x_train, y_train, epochs, learning_rate)
        elif mode == 'mini_batch':
            self.fit_mini_batch(x_train, y_train, epochs, learning_rate, batch_size)
        else:
            self.fit(x_train, y_train, epochs, learning_rate)

        y_train_pred = self.predict(x_train)
        # .flatten() does not work with flatten
        y_train_pred = np.concatenate(y_train_pred)
        # y_train_pred = np.round(y_train_pred).astype(
        #     int)  # classification problem

        y_test_pred = self.predict(x_test)
        # .flatten()   does not work with flatten
        y_test_pred = np.concatenate(y_test_pred)
        # y_test_pred = np.round(y_test_pred).astype(int)   # classification problem

        # Convert one-hot encoded predictions back to class labels
        y_train_pred_labels = np.argmax(y_train_pred, axis=1)
        y_test_pred_labels = np.argmax(y_test_pred, axis=1)

        # Convert one-hot encoded true labels back to class labels
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        # accuracy
        print("#"*50)
        print("Accuracy on train: ", accuracy_score(
            y_true=y_train_labels, y_pred=y_train_pred_labels))
        print("Accuracy on test: ", accuracy_score(
            y_true=y_test_labels, y_pred=y_test_pred_labels))

        print("#"*50)
        print("Balanced Accuracy on train: ", balanced_accuracy_score(
            y_true=y_train_labels, y_pred=y_train_pred_labels))
        print("Balanced Accuracy on test: ", balanced_accuracy_score(
            y_true=y_test_labels, y_pred=y_test_pred_labels))
        # recall
        print("#"*50)
        print("Recall on train: ", recall_score(y_true=y_train_labels,
              y_pred=y_train_pred_labels))
        print("Recall on test: ", recall_score(y_true=y_test_labels,
              y_pred=y_test_pred_labels))
        # precision
        print("#"*50)
        print("Precision on train: ", precision_score(
            y_true=y_train_labels, y_pred=y_train_pred_labels))
        print("Precision on test: ", precision_score(
            y_true=y_test_labels, y_pred=y_test_pred_labels))

        # plot confusion matrices
        print("#"*50)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        c_1 = confusion_matrix(y_true=y_train_labels,
                               y_pred=y_train_pred_labels)
        cmd_1 = ConfusionMatrixDisplay(c_1)
        cmd_1.plot(ax=ax[0], cmap=plt.cm.Blues)
        ax[0].set_title("Confusion matrix: train data")

        c_2 = confusion_matrix(y_true=y_test_labels,
                               y_pred=y_test_pred_labels)
        cmd_2 = ConfusionMatrixDisplay(c_2)
        cmd_2.plot(ax=ax[1], cmap=plt.cm.Blues)
        ax[1].set_title("Confusion matrix: test data")

        plt.tight_layout()
        plt.suptitle("Neural Network: our implementation",
                     fontsize=15, ha='center')
        plt.subplots_adjust(top=0.85)

        plt.show()

    def nn_evaluate_one_hot(self, x_train, y_train, x_test, y_test, epochs, learning_rate, mode=None):
        if mode == 'batch':
            self.fit_batch(x_train, y_train, epochs, learning_rate)
        elif mode == 'ridge':
            self.fit_plus_ridge(x_train, y_train, epochs, learning_rate)
        else:
            self.fit(x_train, y_train, epochs, learning_rate)

        y_train_pred = self.predict(x_train)
        y_train_pred = np.concatenate(y_train_pred)
        y_test_pred = self.predict(x_test)
        y_test_pred = np.concatenate(y_test_pred)

        # Convert one-hot encoded predictions back to class labels
        y_train_pred_labels = np.argmax(y_train_pred, axis=1)
        y_test_pred_labels = np.argmax(y_test_pred, axis=1)

        # Convert one-hot encoded true labels back to class labels
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        # accuracy
        print("#"*50)
        print("Accuracy on train: ", accuracy_score(
            y_true=y_train_labels, y_pred=y_train_pred_labels))
        print("Accuracy on test: ", accuracy_score(
            y_true=y_test_labels, y_pred=y_test_pred_labels))
        # recall
        print("#"*50)
        print("Recall on train: ", recall_score(y_true=y_train_labels,
              y_pred=y_train_pred_labels, average='micro'))
        print("Recall on test: ", recall_score(y_true=y_test_labels,
              y_pred=y_test_pred_labels, average='micro'))
        # precision
        print("#"*50)
        print("Precision on train: ", precision_score(
            y_true=y_train_labels, y_pred=y_train_pred_labels, average='micro'))
        print("Precision on test: ", precision_score(
            y_true=y_test_labels, y_pred=y_test_pred_labels, average='micro'))

        # plot confusion matrices
        print("#"*50)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        c_1 = confusion_matrix(y_true=y_train_labels,
                               y_pred=y_train_pred_labels)
        cmd_1 = ConfusionMatrixDisplay(c_1, display_labels=['0', '1', '2'])
        cmd_1.plot(ax=ax[0], cmap=plt.cm.Blues)
        ax[0].set_title("Confusion matrix: train data")

        c_2 = confusion_matrix(y_true=y_test_labels, y_pred=y_test_pred_labels)
        cmd_2 = ConfusionMatrixDisplay(c_2, display_labels=['0', '1', '2'])
        cmd_2.plot(ax=ax[1], cmap=plt.cm.Blues)
        ax[1].set_title("Confusion matrix: test data")

        plt.tight_layout()
        plt.suptitle("Neural Network: our implementation",
                     fontsize=15, ha='center')
        plt.subplots_adjust(top=0.85)

        plt.show()


# Different activation functions and their derivatives

# for binary classification

# hiperbolical tan
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def softmax_prime(x):
    p = softmax(x)
    return p * (1 - p)


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


# loss function and its derivative


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


# another option for loss function
def categorical_cross_entropy(y_true, y_pred):
    # clipping predicted values, helps to avoid numerical instability
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred))


def categorical_cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true


def binary_cross_entropy(y_true, y_pred):
    # clipping predicted values, helps to avoid numerical instability
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
