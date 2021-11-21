from src.measurements.Measurements import *
from src.preprocessing.insert_nan import nan_percents_str
from src.preprocessing.load_dataset import get_dataset, get_dataset_fully_modified_date
from src.preprocessing.load_dataset import root as root_path
from src.utils.Dataset import save_error
from src.utils.Methods import fill_nan, measures_name, measures
from src.utils.Methods import method_name_single_feature, methods_single_feature
from src.utils.Methods import methods_multiple_feature, method_name_multiple_feature
from src.utils.Methods import method_name_single_feature_param, method_single_feature_param, \
    method_single_feature_param_value

if __name__ == '__main__':

    for nan_percent in nan_percents_str[:3]:
        result_df = pd.DataFrame()
        x, x_nan = get_dataset(nan_percent)
        for method, name in zip(methods_single_feature, method_name_single_feature):
            filled_users = fill_nan(x, x_nan, method)
            temp_result_list = [name, nan_percent]
            for measure, measure_name in zip(measures, measures_name):
                error, error_df = evaluate_dataframe_two(filled_users, measure)
                save_error(error_df, nan_percent, name, measure_name)
                temp_result_list.append(error)
            print("method {} finished".format(name))
            result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)

        for method, name, params in zip(method_single_feature_param, method_name_single_feature_param,
                                        method_single_feature_param_value):
            for param in params:
                filled_users = fill_nan(x, x_nan, method, param)
                temp_result_list = [name + str(param), nan_percent]
                for measure, measure_name in zip(measures, measures_name):
                    error, error_df = evaluate_dataframe_two(filled_users, measure)
                    save_error(error_df, nan_percent, name, measure_name, param)
                    temp_result_list.append(error)
                print("method {} {} finished".format(name, param))
                result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)

        x, x_nan = get_dataset_fully_modified_date(nan_percent)
        for method, name in zip(methods_multiple_feature, method_name_multiple_feature):
            filled_users = fill_nan(x, x_nan, method)
            temp_result_list = [name, nan_percent]
            for measure, measure_name in zip(measures, measures_name):
                error, error_df = evaluate_dataframe_two(filled_users, measure)
                save_error(error_df, nan_percent, name, measure_name)
                temp_result_list.append(error)
            print("method {} finished".format(name))
            result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)
        #
        # x, x_nan = get_dataset_fully_modified_date(nan_percent)
        # for i in range(len(methods_multiple_feature_multi_params)):
        #     for param in methods_multiple_feature_multi_params[i].params:
        #         filled_users = apply_parallel(x_nan.groupby("id"),
        #         methods_multiple_feature_multi_params[i].fill_nan, param)
        #         filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        #         temp_result_list = ["{}_param_{}".format(methods_name_multiple_feature_multi_params[i],
        #         param), nan_percent]
        #         for measure in measures:
        #             measured_value = evaluate_dataframe(filled_users, measure)
        #             temp_result_list.append(measured_value)
        #         result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)
        #     print("method {} finished".format(methods_name_multiple_feature_multi_params[i]))

        # x, x_nan = get_complete_dataset(nan_percent)
        # for i in range(len(methods_complete_feature)):
        #     filled_users = apply_parallel(x_nan.groupby("id"), methods_complete_feature[i].fill_nan)
        #     filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        #     temp_result_list = [method_name_complete_feature[i]]
        #     for measure in measures:
        #         measured_value = evaluate_dataframe(filled_users, measure)
        #         temp_result_list.append(measured_value)
        #     print("method {} finished".format(method_name_complete_feature[i]))
        #     result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)

        result_df.columns = ["Method", "Nan Percent", "Mean Square Error", "Mean Absolute Error",
                             "Mean Absolute Percentage Error"]
        # plot_result(result_df)

        result_df.to_csv(root_path + f"results/methods result_{nan_percent}.csv", index=False)
