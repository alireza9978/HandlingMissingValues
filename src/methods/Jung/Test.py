import numpy as np
import tensorflow as tf
from numpy import argmax
from numpy import array
from numpy import tensordot
from numpy.linalg import norm
from scipy.optimize import differential_evolution
# global optimization to find coefficients for weighted ensemble on blobs problem
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# make an ensemble prediction for multi-class classification

def ensemble_predictions(members, weights, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = array(yhats)
    # weighted sum across ensemble members
    summed = tensordot(yhats, weights, axes=((0), (0)))
    # argmax across classes
    result = argmax(summed, axis=1)
    return result


# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testX, testy):
    # make prediction
    yhat = ensemble_predictions(members, weights, testX)
    # calculate accuracy
    return accuracy_score(testy, yhat)


# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


# loss function for optimization process, designed to be minimized
def loss_function(weights, members, testX, testy):
    # normalize weights
    normalized = normalize(weights)
    # calculate error rate
    return 1.0 - evaluate_ensemble(members, normalized, testX, testy)


if __name__ == '__main__':
    input_shape = 10

    mlp_values = np.array([5, 10, 15, 33, -7, -58, -99, 4, -6, 0]).reshape(-1, 1)
    mlp_result = np.array([-79.3]).reshape(-1, 1)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    mlp_values = x_scaler.fit_transform(mlp_values)
    mlp_result = y_scaler.fit_transform(mlp_result)
    mlp_values = mlp_values.squeeze().reshape(1, -1).repeat(1000, axis=0)
    mlp_result = mlp_result.squeeze().reshape(1, -1).repeat(1000, axis=0)
    mlp_values_dataset = tf.data.Dataset.from_tensor_slices((mlp_values, mlp_result)).batch(32)

    # evaluate averaging ensemble (equal weights)
    weights = [1.0 / input_shape for _ in range(input_shape)]
    score = evaluate_ensemble(members, weights, testX, testy)
    print('Equal Weights Score: %.3f' % score)
    # define bounds on each weight
    bound_w = [(0.0, 1.0) for _ in range(n_members)]
    # arguments to the loss function
    search_arg = (members, testX, testy)
    # global optimization of ensemble weights
    result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
    # get the chosen weights
    weights = normalize(result['x'])
    print('Optimized Weights: %s' % weights)
    # evaluate chosen weights
    score = evaluate_ensemble(members, weights, testX, testy)
    print('Optimized Weights Score: %.3f' % score)
