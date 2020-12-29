# -*- coding: utf-8 -*-
# This Program

import time
import math
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, (None, n_H0, n_W0, n_C0), name="X")
    Y = tf.placeholder(tf.float32, (None, n_y), name="y")

    return X, Y


def initialize_parameters_binary():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", (4, 4, 3, 8), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", (2, 2, 8, 16), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [1, 6], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable("b4", [1, 1], initializer=tf.zeros_initializer())

    parameters = {
        "W1": W1,
        "W2": W2,
        "W4": W4,
        "b4": b4
    }

    return parameters


def L_initialize_parameters_binary(layer_dims):
    tf.set_random_seed(1)
    #layer_dims = [[(4, 4, 3, 8), (2, 2, 8, 16)], [6, 1]]
    conv_layers = layer_dims[0]  # [(4, 4, 3, 8), (2, 2, 8, 16)]
    fc_layers = layer_dims[1]  # [6, 1]

    parameters = {}
    L_conv = len(conv_layers)
    # conv part
    for l in range(L_conv):
        parameters["W" + str(l+1)] = tf.get_variable("W" + str(l+1), conv_layers[l], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    L_full = len(fc_layers)
    # fully connected part
    for l in range(L_full-1):
        parameters["W" + str(l+L_conv+1)] = tf.get_variable("W" + str(l+L_conv+1), [fc_layers[l+1], fc_layers[l]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b" + str(l+L_conv+1)] = tf.get_variable("b" + str(l+L_conv+1), [fc_layers[l+1], 1], initializer=tf.zeros_initializer())

    return parameters


def forward_propagation_binary(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W4 = parameters['W4']
    b4 = parameters['b4']

    # stride = [1, s, s, 1], padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    # filter = ksize = 8 x 8 = [1, f, f, 1], sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
    P2 = tf.contrib.layers.flatten(P2)
    # Z3.shape = (?, 6)
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, tf.transpose(A3)), b4)

    return Z4


def L_forward_propagation_binary(X, parameters, layer_dims, strides,
                                 paddings, pool_filters, pool_strides, pool_paddings):
    conv_layers = layer_dims[0]
    fc_layers = layer_dims[1]

    L_conv = len(conv_layers)
    Pl = X
    # conv part
    for l in range(L_conv):
        Zl = tf.nn.conv2d(Pl, parameters["W" + str(l+1)], strides=[1, strides[l], strides[l], 1], padding=paddings[l])
        Al = tf.nn.relu(Zl)
        Pl = tf.nn.max_pool(Al, ksize=[1, pool_filters[l], pool_filters[l], 1], strides=[1, pool_strides[l], pool_strides[l], 1], padding=pool_paddings[l])

    Al = tf.contrib.layers.flatten(Pl)

    L_fc = len(fc_layers)
    # fully connected part
    Zl = tf.contrib.layers.fully_connected(Al, fc_layers[0], activation_fn=None)
    for l in range(L_fc-1):
        Al = tf.nn.relu(Zl)
        Zl = tf.transpose(tf.add(tf.matmul(parameters["W" + str(l+L_conv+1)], tf.transpose(Al)), parameters["b" + str(l+L_conv+1)]))

    return tf.transpose(Zl)


def compute_cost_binary(Z4, Y):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z4, labels=tf.transpose(Y)))

    return cost


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # suffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # partition except end case
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # end case
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def load_data(filename):
    """
    Dataset format should is h5 and look like this:
    datasets/
    --filename
    ----train_x (m, 64, 64, 3)
    ----train_y (m,)

    m := number of training examples
    filename := e.g. "mydataset.h5"
    """
    dataset = h5py.File(str(filename), "r")
    X = np.array(dataset["train_x"][:])
    Y = np.array(dataset["train_y"][:])
    # reshape from (m,) to (1, m)
    Y = Y.reshape((1, Y.shape[0]))

    return X, Y


def preprocess(X):
    # The "-1" makes reshape flatten the remaining dimensions
    X_flatten = X.reshape(X.shape[0], -1).T
    # Standardize data to have values between 0 and 1
    return X_flatten / 255.


def three_layer_ConvNet_tf(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
                           num_epochs=80, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters_binary()

    Z4 = forward_propagation_binary(X, parameters)

    cost = compute_cost_binary(Z4, Y)

    # backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # compute tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        # training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                # run session to execute the optimizer
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # calculate correct predictions
        # predict_op = tf.argmax(Z3, 1)
        # correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        #print('Z4: ' + str(tf.round(tf.sigmoid(Z4)).eval({X: X_train, Y: Y_train})))
        #print('Y.shape: ' + str(np.array(Z4.eval({X: X_train, Y: Y_train})).shape))
        #print('Y: ' + str(Y.eval({X: X_train, Y: Y_train})))
        correct_prediction = tf.equal(tf.round(tf.sigmoid(tf.transpose(Z4))), Y)
        #print('correct_prediction: ' + str(correct_prediction.eval({X: X_train, Y: Y_train})))

        # train/test accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


def L_layer_ConvNet_tf(X_train, Y_train, X_test, Y_test, layer_dims,
                       strides, paddings, pool_filters, pool_strides,
                       pool_paddings, learning_rate=0.009, num_epochs=80,
                       minibatch_size=64, print_cost=True):


    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = L_initialize_parameters_binary(layer_dims)

    Z4 = L_forward_propagation_binary(X, parameters, layer_dims, strides,
                                    paddings, pool_filters, pool_strides, pool_paddings)

    cost = compute_cost_binary(Z4, Y)

    # backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # compute tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        # training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                # run session to execute the optimizer
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # calculate correct predictions
        # predict_op = tf.argmax(Z3, 1)
        # correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        #print('Z4: ' + str(tf.round(tf.sigmoid(Z4)).eval({X: X_train, Y: Y_train})))
        #print('Y.shape: ' + str(np.array(Z4.eval({X: X_train, Y: Y_train})).shape))
        #print('Y: ' + str(Y.eval({X: X_train, Y: Y_train})))
        correct_prediction = tf.equal(tf.round(tf.sigmoid(tf.transpose(Z4))), Y)
        #print('correct_prediction: ' + str(correct_prediction.eval({X: X_train, Y: Y_train})))

        # train/test accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


def main():
    print(time.time())

    filename = "../datasets/catvnoncat_2.h5"
    X_train, Y_train = load_data(filename)
    #X_train = preprocess(X_train)
    X_train = X_train / 255.
    Y_train = Y_train.T
    #print("X_train.shape: ", X_train.shape)
    #print("Y_train.shape: ", Y_train.shape)

    test_filename = "../datasets/train_catvnoncat.h5"
    test_dataset = h5py.File(test_filename, "r")
    X_test = np.array(test_dataset["train_set_x"][:])
    Y_test = np.array(test_dataset["train_set_y"][:])
    # reshape from (m,) to (1, m)
    Y_test = Y_test.reshape((1, Y_test.shape[0]))
    #X_test = preprocess(X_test)
    X_test = X_test / 255.
    Y_test = Y_test.T
    #print("X_test.shape: ", X_test.shape)
    #print("Y_test.shape: ", Y_test.shape)

    # (4, 4, 3, 8) = (filter height, filter width, channel, number of filters)
    layer_dims = [[(3, 3, 3, 4), (4, 4, 4, 8), (2, 2, 8, 16)], [32, 16, 8, 1]]
    strides = [1, 1, 1]
    paddings = ["SAME", "SAME", "SAME"]
    pool_filters = [4, 4, 4]
    pool_strides = [2, 4, 4]
    pool_paddings = ["SAME", "SAME", "SAME"]
    learning_rate = 0.0001
    num_epochs = 50
    minibatch_size = 64
    print_cost = True

    start = time.time()
    print("Start time: ", start)
    # execute model
    params = L_layer_ConvNet_tf(X_train, Y_train, X_test, Y_test,
                                layer_dims, strides, paddings,
                                pool_filters, pool_strides, pool_paddings,
                                learning_rate, num_epochs, minibatch_size, print_cost)
    
    """
    thre_layer_ConvNet_tf
    Train Accuracy: 0.5821256
    Test Accuracy: 0.6363636
    Time:  85.18749856948853
    L_layer_ConvNet_tf
    Train Accuracy: 0.5821256
    Test Accuracy: 0.6363636
    Time:  34.90772294998169
    Train Accuracy: 0.5821256
    Test Accuracy: 0.6363636
    Time:  47.118125677108765
    """

    diff = time.time() - start
    print("Time: ", diff)


if __name__ == "__main__":
    main()
#
#
#
#
#
#
#
#
#
#
