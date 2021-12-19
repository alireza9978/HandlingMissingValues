from src.preprocessing.air.load_dataset import load_air_dfs
from src.utils.Methods import methods_trainable

if __name__ == '__main__':

    data_frames = load_air_dfs()
    for model in methods_trainable:
        temp = model(data_frames)
        temp.train_test_save("air")
        print(f"model {temp.get_name()} finished")
        del temp
