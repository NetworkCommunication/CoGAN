import csv
import os
import re
import matplotlib as mpl
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.model_zoo import tqdm
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.cuda import device

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from model.model import VAE, Discriminator

def load_data1():
    splits1 = ["test0", "test3"]
    train_data_path0 = "D:/TEST0/"  # ./data2/train_data_v2/
    train_data_path1 = "D:/TEST3/"
    shuffle1 = {'test0': False, 'test3': False}
    dataset = {}
    train_file_list0 = os.listdir(train_data_path0)
    train_file_list1 = os.listdir(train_data_path1)

    pattern = re.compile(r'TEST_(\d+)')
    pattern1 = re.compile(r'data_(\d+)')
    train_file_list0 = sorted(train_file_list0, key=lambda x: int(pattern.search(x).group(1)))
    train_file_list1 = sorted(train_file_list1, key=lambda x: int(pattern1.search(x).group(1)))

    test0_data,test1_data = [],[]
    for obj0 in train_file_list0:
        train_file_path0 = train_data_path0 + obj0
        train_matrix0= np.load(train_file_path0)
        test0_data.append(train_matrix0)
    for obj1 in train_file_list1:
        train_file_path1 = train_data_path1 + obj1
        train_matrix1 = np.load(train_file_path1)
        test1_data.append(train_matrix1)

    dataset["test0"] = torch.from_numpy(np.array(test0_data)).float()
    dataset["test3"] = torch.from_numpy(np.array(test1_data)).float()

    dataloader0 = {x: torch.utils.data.DataLoader(
                                dataset=dataset[x], batch_size=256, shuffle=shuffle1[x])  # shuffle=shuffle
                                for x in splits1}
    return     dataloader0
def test(G, disc, dataloader_test0,dataloader_test3, kappa=1.0):
    G.load_state_dict(torch.load("D:/CoGAN/checkpoints/model1.pth"))
    disc.load_state_dict(torch.load("D:/CoGAN/checkpoints/model3.pth"))
    G.to(device).eval()
    disc.to(device).eval()
    criterion = nn.MSELoss()

    i = -1
    dataLoaders = [dataloader_test0, dataloader_test3]
    for dataLoader in dataLoaders:
        for inputs in dataLoader:
            inputs = inputs
            inputs = inputs.permute(0,2,1,3,4)
            outputs2, z3, z4, KL1, KL2 = G(inputs)
            img_distance = criterion( outputs2 , inputs)
            anomaly_score = img_distance
            z_distance = criterion(z3, z4)



if __name__ == "__main__":
    dataLoader1 = load_data1()
    G = VAE()
    disc = Discriminator()
    test(G, disc, dataLoader1["test0"], dataLoader1["test3"])
    df = pd.read_csv("data/score.csv")
    trainig_label = 0
    labels = np.where(df["label"].values == trainig_label, 0, 1)
    anomaly_score = df["anomaly_score"].values
    fpr, tpr, a = roc_curve(labels,  anomaly_score )
    precision, recall, b = precision_recall_curve(labels,  anomaly_score )
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    file.close()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC-AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    plt.plot(recall, precision, label=f"PR = {pr_auc:3f}")
    plt.title("PR-AUC")
    plt.xlabel("Recall")
    plt.ylabel("Pecision")
    plt.legend()
    plt.show()





