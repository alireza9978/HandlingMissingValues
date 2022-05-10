import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from numpy.linalg import norm
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.methods.Jung.SmallNeuralNet import SmallNeuralNet
from src.preprocessing.smart_star.load_dataset import get_dataset_fully_modified_date, root

class MultiLayerPerceptron:

    def __init__(self, temp_df: pd.DataFrame, user_id, small_neural_net_count,
                 small_neural_net_training_epochs, training_epochs):
        self.user_id = user_id
        self.history = []
        self.small_neural_net_count = small_neural_net_count
        self.small_neural_net_training_epochs = small_neural_net_training_epochs
        self.training_epoch = training_epochs
        self.x_scaler = MultiLayerPerceptron._generate_scaler()
        self.y_scaler = MultiLayerPerceptron._generate_scaler()
        self._preprocess_data_set(temp_df)
        self._generate_model(self.small_neural_net_count)
        self.small_set_predictions = None
        self.weights = None

    @staticmethod
    def _generate_scaler():
        return MinMaxScaler()

    def _preprocess_data_set(self, temp_df: pd.DataFrame):
        self.nan_indexes = temp_df.index[temp_df.usage.isna()].to_numpy()
        temp_nan_df = temp_df[temp_df.usage.isna()]
        temp_df = temp_df[~temp_df.usage.isna()]
        y_columns = ["usage"]
        useless_columns = ["id"]

        train_x = temp_df.drop(columns=useless_columns + y_columns)
        test_x = temp_nan_df.drop(columns=useless_columns + y_columns)
        temp_columns = train_x.columns
        train_x = train_x.to_numpy()
        test_x = test_x.to_numpy()

        train_y = temp_df[y_columns].to_numpy()

        train_x = self.x_scaler.fit_transform(train_x)
        test_x = self.x_scaler.transform(test_x)
        train_y = self.y_scaler.fit_transform(train_y)

        self.dataset = pd.DataFrame(train_x, columns=temp_columns)
        self.dataset[y_columns] = train_y

        self.train_x_dataset = train_x
        self.train_y_dataset = train_y

        self.test_dataset = test_x

        self._save_scaler()

    def _save_scaler(self):
        dump(self.x_scaler, root + f'saved_models/neural_net_scaler_x_{self.user_id}.joblib')
        dump(self.y_scaler, root + f'saved_models/neural_net_scaler_y_{self.user_id}.joblib')

    def _generate_model(self, input_shape):
        self.small_neural_nets = []
        for i in range(input_shape):
            self.small_neural_nets.append(SmallNeuralNet(self.dataset, self.x_scaler,
                                                         self.y_scaler, self.small_neural_net_training_epochs))

    # normalize a vector to have unit norm
    @staticmethod
    def normalize(weights):
        # calculate l1 vector norm
        result = norm(weights, 1)
        # check for a vector of all zeros
        if result == 0.0:
            return weights
        # return normalized vector (unit norm)
        return weights / result

    def ensemble_predictions(self):
        return MultiLayerPerceptron.static_ensemble_predictions(self.train_x_dataset, self.small_neural_nets)

    @staticmethod
    def static_ensemble_predictions(dataset, models):
        # make predictions
        y_hats = [model.predict(dataset) for model in models]
        y_hats = np.concatenate(y_hats, axis=1)
        return y_hats

    @staticmethod
    def apply_weights(y_hat, weights):
        y_hat = (y_hat * weights).mean(axis=1).reshape(-1, 1)
        return y_hat

    @staticmethod
    def static_evaluate_ensemble(y_hat, weights, train_y_dataset):
        y_hat = MultiLayerPerceptron.apply_weights(y_hat, weights)
        return mean_squared_error(train_y_dataset, y_hat)

    # # evaluate a specific number of members in an ensemble
    def evaluate_ensemble(self, weights):
        return MultiLayerPerceptron.static_evaluate_ensemble(self.small_set_predictions, weights, self.train_y_dataset)

    # loss function for optimization process, designed to be minimized
    @staticmethod
    def loss_function(weights, self):
        # normalize weights
        normalized = MultiLayerPerceptron.normalize(weights)
        # calculate error rate
        loss = self.evaluate_ensemble(normalized)
        self.history.append(loss)
        return loss

    def train_model(self):
        good_small_neural_net_threshold = 1 / (self.small_neural_net_count * 2)
        maximum_loss = 0.2
        maximum_iteration = self.training_epoch
        current_iteration = 1
        bound_weights = [(0.0, 1.0) for _ in range(self.small_neural_net_count)]
        for small_model in self.small_neural_nets:
            small_model.train_model()
        self.small_set_predictions = self.ensemble_predictions()
        print("start")
        while True:
            result = differential_evolution(MultiLayerPerceptron.loss_function, bound_weights, [self], maxiter=500,
                                            tol=1e-7)
            print(result['message'], " number of iterations = ", result['nit'])
            # get the chosen weights
            weights = MultiLayerPerceptron.normalize(result['x'])
            temp_loss = result['fun']

            if current_iteration >= maximum_iteration:
                self.weights = weights
                print("did not converge")
                break
            bad_small_neural_nets_count = (weights < good_small_neural_net_threshold).sum()
            if bad_small_neural_nets_count == 0:
                if temp_loss < maximum_loss:
                    self.weights = weights
                    print("converged")
                    break

            for i in sorted(np.where(weights < good_small_neural_net_threshold)[0].tolist(), reverse=True):
                self.small_neural_nets.remove(self.small_neural_nets[i])
            for i in range(bad_small_neural_nets_count):
                new_model = SmallNeuralNet(self.dataset, self.x_scaler, self.y_scaler,
                                           self.small_neural_net_training_epochs)
                new_model.train_model()
                self.small_neural_nets.append(new_model)

            print(current_iteration)
            current_iteration += 1

    def save_plot(self):
        if self.history is not None:
            plt.plot(self.history, label='train_loss')
            plt.legend()
            plt.savefig(root + f"results/Jung/multi_layer_perceptron/{self.user_id}.jpeg")

    def predict_nan(self):
        if self.weights is None:
            print("First, train the model")
            return None
        y_pred = MultiLayerPerceptron.static_ensemble_predictions(self.test_dataset, self.small_neural_nets)
        y_pred = MultiLayerPerceptron.apply_weights(y_pred, self.weights)
        return self.y_scaler.inverse_transform(y_pred)


def fill_nan(temp_df: pd.DataFrame):
    user_id = temp_df["id"].values[0]
    temp_model = MultiLayerPerceptron(temp_df, user_id, 10, 20, 10)
    temp_model.train_model()
    temp_model.save_plot()
    return pd.Series([temp_model.predict_nan(), temp_model.nan_indexes])


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.05")
    x = x[x.id == 45]
    x_nan = x_nan[x_nan.id == 45]
    filled_users = x_nan.groupby("id").apply(fill_nan)
    # filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
