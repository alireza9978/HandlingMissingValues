import numpy as np
from sklearn.metrics import mean_squared_error as rmse

from fcm_estimator import FCMeansEstimator
from genetic_algorithm import GeneticAlgorithm
from src.methods.Aydilek2013.svr_estimator import SVREstimator
# Print the results and RMSE value
from src.preprocessing.load_dataset import get_dataset


def show_results(true_values, predicted_values):
    print('Algorithm finished.\n')
    for i in range(len(true_values)):
        print('True value ' + '{:02x}'.format(i + 1) + ': ' + '{:0.5f}'.format(true_values[i]) + '\t' +
              'Estimated value ' + '{:02x}'.format(i + 1) + ': ' + '{:0.5f}'.format(predicted_values[i]))

    print('\nRMSE  : ' + str(rmse(true_values, predicted_values)))


if __name__ == '__main__':
    data, incomplete_data = get_dataset()

    # Make FCM and SVR model
    fcm_estimator = FCMeansEstimator(c=3, m=3, data=incomplete_data)
    svr_estimator = SVREstimator(data=incomplete_data)
    y = svr_estimator.estimate_missing_value()

    while True:
        ga = GeneticAlgorithm(fcm_estimator, svr_estimator)
        c, m = ga.run()
        fcm_estimator = FCMeansEstimator(c=c, m=m, data=incomplete_data)
        x = fcm_estimator.estimate_missing_values()
        error = np.power(x - y, 2).sum()
        if error < 1:
            predicted_values = x
            break

    rows = fcm_estimator.incomplete_rows
    true_values = []
    for i in rows:
        true_values.append(data[i][np.where(np.isnan(incomplete_data[i]))[0][0]])

    show_results(true_values, predicted_values)
