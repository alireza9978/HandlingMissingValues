import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

if __name__ == '__main__':
    x = np.array([1, 1, 1, 1, 1.25, 1.5, 1.75, 2, 1.75, 1.5, 1.25, 1, 0.5, 0, 0.5, 1, 1.5, 2, 2.5, 2, 1, 0.5, 0, 0])
    x_nan = np.array([1, 1, 1, 1, 1.25, np.nan, 1.75, 2, 1.75, np.nan, 1.25, 0.75,
                      0.5, 0.5, 0.5, 1, np.nan, np.nan, np.nan, np.nan, 1, 0.5, 0, 0])

    temp_df = pd.DataFrame(x_nan)
    indexes = np.array(list(range(x.shape[0])))
    nan_index = np.isnan(x_nan)
    not_nan_index = ~np.isnan(x_nan)
    x_train = indexes[not_nan_index].reshape(-1, 1)
    y_train = x[not_nan_index].reshape(-1, 1)
    x_test = indexes.reshape(-1, 1)

    plt.plot(indexes, x, label="real x")
    plt.plot(indexes[np.isnan(x_nan)], x[np.isnan(x_nan)], label="nan", marker="o", linestyle='None')

    # reg = LinearRegression()
    # reg.fit(x_train, y_train)
    # pred = reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="linear_regression")
    #
    # degree = 2
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_2")
    #
    # degree = 3
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_3")
    #
    # degree = 4
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_4")
    #
    # degree = 5
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_5")
    #
    # degree = 6
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_6")
    #
    degree = 7
    polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polynomial_reg.fit(x_train, y_train)
    pred = polynomial_reg.predict(x_test).squeeze()
    plt.plot(indexes, pred, label="polynomial_regression_7")

    degree = 8
    polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polynomial_reg.fit(x_train, y_train)
    pred = polynomial_reg.predict(x_test).squeeze()
    plt.plot(indexes, pred, label="polynomial_regression_8")

    # degree = 9
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_9")
    #
    # degree = 10
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_10")
    #
    # degree = 11
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_11")
    #
    # degree = 12
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_12")

    degree = 13
    polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polynomial_reg.fit(x_train, y_train)
    pred = polynomial_reg.predict(x_test).squeeze()
    plt.plot(indexes, pred, label="polynomial_regression_13")
    #
    # degree = 14
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_14")

    # degree = 15
    # polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # polynomial_reg.fit(x_train, y_train)
    # pred = polynomial_reg.predict(x_test).squeeze()
    # plt.plot(indexes, pred, label="polynomial_regression_15")

    # filled_x = temp_df.interpolate(method="polynomial", order=2).to_numpy().squeeze()
    # plt.plot(indexes[np.isnan(x_nan)], filled_x[np.isnan(x_nan)], label="polynomial_2", marker="x", linestyle='None')
    # filled_x = temp_df.interpolate(method="polynomial", order=3).to_numpy().squeeze()
    # plt.plot(indexes[np.isnan(x_nan)], filled_x[np.isnan(x_nan)], label="polynomial_3", marker="x", linestyle='None')
    # filled_x = temp_df.interpolate(method="quadratic").to_numpy().squeeze()
    # plt.plot(indexes[np.isnan(x_nan)], filled_x[np.isnan(x_nan)], label="quadratic", marker="x", linestyle='None')
    # filled_x = temp_df.interpolate(method="spline",  order=1).to_numpy().squeeze()
    # plt.plot(indexes[np.isnan(x_nan)], filled_x[np.isnan(x_nan)], label="spline", marker="x", linestyle='None')
    # filled_x = temp_df.interpolate(method="spline",  order=2).to_numpy().squeeze()
    # plt.plot(indexes[np.isnan(x_nan)], filled_x[np.isnan(x_nan)], label="spline_2", marker="x", linestyle='None')
    # filled_x = temp_df.interpolate(method="spline",  order=3).to_numpy().squeeze()
    # plt.plot(indexes[np.isnan(x_nan)], filled_x[np.isnan(x_nan)], label="spline_3", marker="x", linestyle='None')
    # filled_x = temp_df.interpolate(method="spline",  order=4).to_numpy().squeeze()
    # plt.plot(indexes[np.isnan(x_nan)], filled_x[np.isnan(x_nan)], label="spline_4", marker="x", linestyle='None')

    plt.legend()
    plt.show()
