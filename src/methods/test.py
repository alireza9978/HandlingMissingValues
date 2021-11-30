from src.preprocessing.insert_nan import nan_percents_str
from src.preprocessing.load_dataset import get_train_test_dataset
from src.utils.Methods import methods_trainable

if __name__ == '__main__':

    for nan_percent in nan_percents_str[:3]:
        data_frames = get_train_test_dataset(nan_percent, 0.3)
        for model in methods_trainable:
            temp = model(data_frames)
            temp.train_test_save(nan_percent)
