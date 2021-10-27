<h1 align = "center"> Handling Missing Values </h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://forthebadge.com/images/badges/made-with-python.svg" alt="python"></a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="python version"></a>
</p>
The goal of our project is to test different existing missing value imputation techniques for time-series and find the best ones. To this goal we implemented many different imputation methods and tested them on the Smart* dataset. This dataset contains the power consumption data for 114 single family apartments.


As the original dataset does not have missing values, we randomly injected np.nan values into the dataset and ran the implemented methods on them. To evaluate the methods, we compared the imputed values with the original values in the datset and calculated the difference between them.
