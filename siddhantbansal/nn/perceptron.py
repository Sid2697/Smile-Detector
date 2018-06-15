# importing necessary packages
import numpy as np


class Perceptron:

    def __init__(self, N, alpha=0.1):
        # initialize the weight matrix and store the learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # apply rh step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert a column of 1's as the last entry in the feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual date point
            for (x, target) in zip(X, y):
                # take the dot product btw the intput features
                # and the weight matrix, then pass this value
                # through the stop function to obtain the prediction
                p = self.step(x.dot(self.W))

                # only perform a weight update if our prediction
                # does not match the target
                if p != target:
                    # determine the error
                    error = p - target

                    # update the weight matrix
                    self.W += -self.alpha * error * x
        # print(self.W)

    def predict(self, X, addBias=True):
        # ensure our input is a matrix
        X = np.atleast_2d(X)

        # check to see if bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the
            # feature matrix
            X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product between input features and the
        # weight matrix, then pass the value through the step function
        return self.step(np.dot(X, self.W))
