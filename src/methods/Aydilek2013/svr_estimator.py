import numpy as np
from sklearn.svm import SVR

from config import SVRParam


class SVREstimator:
    def __init__(self, data):
        self.data = data
        self.complete_rows, self.incomplete_rows = self.__extract_rows()

    # Extract complete and incomplete rows
    def __extract_rows(self):
        nan_state = np.isnan(self.data)
        complete_rows = np.where(~nan_state)[0]
        incomplete_rows = np.where(nan_state)[0]
        return complete_rows, incomplete_rows

    # Estimate the missing values
    def estimate_missing_value(self):
        estimated_data = np.zeros(self.incomplete_rows.shape[0])
        complete_data = self.data[self.complete_rows].copy()
        incomplete_data = self.data[self.incomplete_rows].copy()

        for column, value in enumerate(incomplete_data.transpose()):
            ind_rows = np.where(np.isnan(value))[0]
            if len(ind_rows) > 0:
                # TRAIN IS EMPTY
                x_train = np.delete(complete_data.transpose(), column, 0).transpose()
                y_train = np.array(complete_data[:, column])

                model = SVR(gamma='scale', C=SVRParam.C, epsilon=SVRParam.EP)
                model.fit(x_train, y_train)

                x_test = []
                x_test_temp = np.delete(incomplete_data.transpose(), column, 0).transpose()
                for i in ind_rows:
                    x_test.append(x_test_temp[i])

                predicted = model.predict(np.array(x_test))

                for i, v in enumerate(ind_rows):
                    estimated_data[v] = predicted[i]

        return estimated_data
