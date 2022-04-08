import torch
import torch.nn  as nn
import torch.nn.functional  as F
import torch.optim  as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import json
import monai
from monai.metrics import *
import os
from tqdm import tqdm
from Loader.loader_CT import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from model.load_model import *
from torch.autograd import Variable
from sklearn.metrics import *
import argparse
from torch.optim import *
import logging
import os
import sys
import tempfile
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn import metrics
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, ImageDataset
# from Loader.loader_CT_test import read_data_ds as read_data_ds_test

def calculate_metric(gt, pred):
    pred = np.array(pred)
    pred[pred>0.5]=1
    pred[pred<1]=0
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    acc= (TP+TN)/float(TP+TN+FP+FN)
    sen = TP / float(TP+FN)
    spe=TN / float(TN+FP)
    return acc,sen,spe



def train(parameters):
    # 创建文件夹

    # GPU
    device = parameters["device_type"]
    batch_size = parameters["batch_size"]
    structure = parameters["structure"]
    num_epoch = parameters["max_epochs"]
    continue_training = parameters["continue_training"]
    data_path = parameters["json_path"]
    learning_rate = parameters["learning_rate"]
    pretrain_path =parameters["pretrain_path"]
    best_auc=0

    _, test_ds = read_data_ds(data_path, section='test')

    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    out_path = './test_results/' + structure + '/' + time_save + '/'

    load_path = './training_results/' + structure + '/'


    if not os.path.exists(out_path):
        os.makedirs(out_path)
    net = load_model(structure)

    print(net)
    print('loading model............')
    net.load_state_dict(torch.load(load_path + 'bestmodel.pth'))
    for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
        print(layer, param)

    print(net)
    net.to(device)
    net.eval()
    print('loading model............')
    dict_all={"auc":[],'sen':[],'spe':[],'acc':[]}
    test_ds = train_ds

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False, num_workers=2)


    pre_list = []
    label_list = []
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for val_data in test_bar:
            inputs_RD = val_data['RD'].cuda()
            inputs_CT = val_data['CT'].cuda()

            inputs = torch.cat((inputs_CT, inputs_RD), dim=1)
            labels = val_data['Label'].cuda()
            val_outputs = net(inputs)
            val_outputs=torch.sigmoid(val_outputs)

            val_outputs = torch.squeeze(val_outputs)
            labels = torch.squeeze(labels)
            try:
                pre_list += val_outputs.tolist()
                label_list += labels.tolist()
            except:
                pre_list.append(val_outputs.item())
                label_list.append(labels.item())

        acc,sen,spe = calculate_metric(label_list,pre_list)
        print("acc,sen,spe",acc,sen,spe)
        AUC = metrics.roc_auc_score(label_list, pre_list)
        print("Test AUC", AUC)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters", type=str, default='Img_classification_USA.json', help="parameters json")
    args = parser.parse_args()

    para_path = './parameters/' + args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)