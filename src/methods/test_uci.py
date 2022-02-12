from src.preprocessing.load_dataset import root
from src.preprocessing.uci.clean_dataset import load_dataset
from src.methods.Simple.Interpolation import Interpolation
from src.methods.MovingWindow.MovingMean import MovingMean

if __name__ == '__main__':

    print("start")
    data_frame = load_dataset()
    base_columns = ["id"]
    target_file_path = root + "datasets/power/uci/result_dataset_{}_{}.csv"
    for model in [MovingMean, Interpolation]:
        for train_param in model.get_train_params():
            filled_data_frame = data_frame.copy()
            for col in filled_data_frame.columns[:-1]:
                temp_data_frame = filled_data_frame[base_columns + [col]]
                temp_data_frame.rename(columns={col: "usage"}, inplace=True)
                filled_value, user_id, _ = model.fill_nan(temp_data_frame, train_param)
                filled_data_frame.loc[filled_value.index, col] = filled_value["predicted_usage"]
            filled_data_frame.drop(columns=["id"], inplace=True)
            filled_data_frame.to_csv(target_file_path.format(model.get_name(), train_param), index_label="date")
