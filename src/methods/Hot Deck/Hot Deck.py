import numpy as np

from src.measurements.Measurements import mean_square_error
from src.preprocessing.load_dataset import get_dataset_with_modified_date


def fill_nan(temp_array: np.ndarray):
    print(temp_array)

    # for idx, v in enumerate(missing):  # For each missing element in data set
    #     i, j = v
    #     euclidean = []
    #     euclideanTotal = 0
    #     for r in range(len(importedNM)):  # Loop all non-missing rows...
    #         for c in range(col):  # ...and all of its columns...
    #             if c != j:  # ...except itself...
    #                 euclideanTotal += (imported[i][c] - importedNM[r][
    #                     c]) ** 2  # ...to calculate the euclidean distance of both...
    #         e = math.sqrt(euclideanTotal)
    #         euclidean.append(
    #             [e, importedNM_index[r]])  # Append found euclidean and index of that in the original data set
    #     sorted(euclidean, key=lambda l: l[0], reverse=True)  # Sorts the euclidean list by their first value
    #     lst = [imported[euclidean[r][1]][j] for r in range(kHD)]  # Gets the list of first kHD elements of those values
    #     imported[i][j] = Counter(lst).most_common(1)[0][0]  # Imputes the most common element from above list.
    #     printProgress(idx + 1, miss, v, imported[i][j])

    return None


if __name__ == '__main__':
    x, x_nan = get_dataset_with_modified_date()
    filled_x, nan_index = x.groupby("id").apply(fill_nan(x_nan))
    print(mean_square_error(x[nan_index], filled_x))
