from src.preprocessing.insert_nan import nan_percents_str
from src.preprocessing.load_dataset import get_train_test_dataset, get_train_test_fully_modified_date, \
    get_train_test_dataset_triple
from src.utils.Methods import methods_trainable, methods_trainable_modified_dataset

if __name__ == '__main__':

    for nan_percent in nan_percents_str[:1]:

        data_frames_list = get_train_test_dataset_triple(nan_percent)
        for data_frames in data_frames_list:
            print(data_frames)
            # for model in methods_trainable:
            #     temp = model(data_frames)
            #     temp.train_test_save(nan_percent)
            #     print(f"model {temp.get_name()} finished")
            #     del temp

        # data_frames = get_train_test_fully_modified_date(nan_percent, 0.3)
        # for model in methods_trainable_modified_dataset:
        #     temp = model(data_frames)
        #     temp.train_test_save(nan_percent)
        #     print(f"model {temp.get_name()} finished")
        #     del temp
