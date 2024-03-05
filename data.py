import random
import torch
import numpy as np
import os
import os
import re
from sklearn.model_selection import train_test_split

splits = ["train", "test0", "test3"]
train_data_path = "D:/train_data_v1/"
shuffle = {'train': False, 'test0': False, 'test3': False}
def load_data():
    dataset = {}
    train_file_list = os.listdir(train_data_path)
    pattern = re.compile(r"(?<=_)\d+(?=_\d+\.npy)")

    train_file_list = sorted(train_file_list,
                             key=lambda x: (int(pattern.search(x).group()), int(x.split("_")[-1].split(".")[0])))
    train_data_count = {}
    for file_name in train_file_list:
        train_data = file_name.split("_")[0]
        if train_data not in train_data_count:
            train_data_count[train_data] = 1
        else:
            train_data_count[train_data] += 1
    for train_data, count in train_data_count.items():
        print(f"{train_data}: {count}")


    train_data_count = {}
    for file_name in train_file_list:
        train_data_x = file_name.split("_")[2]
        if train_data_x not in train_data_count:
            train_data_count[train_data_x] = 1
        else:
            train_data_count[train_data_x] += 1

    for train_data_x, count in train_data_count.items():
        print(f"{train_data_x}: {count}个.npy文件")
    small_train_data_count = 0
    for train_data_x, count in train_data_count.items():
        if count <= 0:
            small_train_data_count += 1
    large_train_data = []
    for train_data_x, count in train_data_count.items():

        if count > 0:
            large_train_data.append(train_data_x)
    train_file_list1 = []
    for file_name in train_file_list:
        train_data_x = file_name.split("_")[2]
        if train_data_x in large_train_data:
            train_file_list1.append(file_name)
    train_data = []
    for obj in train_file_list1:
        train_file_path = train_data_path + obj
        train_matrix = np.load(train_file_path,allow_pickle=True)
        train_data.append(train_matrix)

    train_size = len(train_data)
    indices = list(range(train_size))
    train_idx = indices[:int(.94 * train_size)]
    test_idx = indices[int(.94 * train_size):]
    dataset["train"] = torch.from_numpy(np.array(train_data)[train_idx]).float()
    dataset["test0"] = torch.from_numpy(np.array(train_data)[test_idx]).float()
    dataset["test3"] = dataset["test0"][ : ]
    dataset["test0"] = dataset["test0"][:]
    test_3 = []
    i =3
    for train_matrix3 in dataset["test3"]:
        train_matrix3 =np.array(train_matrix3)
        if i ==3:
            noise = np.random.normal(loc=0, scale=3, size=(train_matrix3.shape[0], 1, 1, train_matrix3.shape[3]))
            noise +=3
            train_matrix3[:, 0, 0, :] += noise[:, 0, 0, :]
            test_3.append(train_matrix3)
    dataset["test3"] = torch.from_numpy(np.array(test_3)).float()
    dataloader = {x: torch.utils.data.DataLoader(
                                dataset=dataset[x], batch_size=256, shuffle=shuffle[x])
                                for x in splits}
    return     dataloader


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    dataLoader = load_data()















