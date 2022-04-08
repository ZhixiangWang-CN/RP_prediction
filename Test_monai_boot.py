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
    # print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    # print('Sensitivity:',TP / float(TP+FN))
    # print('Specificity:',TN / float(TN+FP))
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
    # train_ds,val_ds=read_data_ds(data_path)
    # _,test_ds = read_data_ds(data_path,section='training')
    # _, inde_test_ds = read_data_ds(data_path, section='validation')
    # _, inde_test_ds = read_data_ds(data_path, section='training')
    _, train_ds = read_data_ds(data_path, section='training')

    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    out_path = './test_results/' + structure + '/' + time_save + '/'

    load_path = './training_results/' + structure + '/'

    # 定义判别器  #####Discriminator######使用多层网络来作为判别器
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    net = load_model(structure)
    fc_wight_list=[]
    print(net)
    print('loading model............')
    # load_path='C:/Softwares/Codes/DL_img_classification/Trained_results/Classification/AUC0.83resnet10848484/2022_03_04_09_52/'
    # load_path = 'C:/Softwares/Codes/DL_img_classification/Trained_results/Classification/AUC0.87/2022_03_16_16_26/'
    net.load_state_dict(torch.load(load_path + 'bestmodel.pth'))
    for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
        print(layer, param)
        if 'classififcation.5.weight' == layer:
            fc_wight_list.append(param)
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)

    # load_path = 'C:/Softwares/Codes/DL_img_classification/Trained_results/Classification/AUC0.83_0327_84/2022_03_27_12_17/'
    # net = load_model(structure)
    # net.load_state_dict(torch.load(load_path + 'bestmodel24.pth'))
    for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
        print(layer, param)
        if 'classififcation.5.weight' == layer:
            fc_wight_list.append(param)
    if len(fc_wight_list) > 1:
        c = fc_wight_list[len(fc_wight_list) - 1] - fc_wight_list[len(fc_wight_list) - 2]
        summ = torch.sum(c)
        print("FC 差值", summ)
    print(net)
    net.to(device)
    net.eval()
    print('loading model............')
    dict_all={"auc":[],'sen':[],'spe':[],'acc':[]}
    test_ds = train_ds
    # for o in range(4):
    #     boot_data = np.random.choice(train_ds.data, len(train_ds.data), replace=True)
    #     test_ds.data = boot_data
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False, num_workers=2)


    # print("Test",o)
    pre_list = []
    label_list = []
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for val_data in test_bar:
            inputs_RD = val_data['RD'].cuda()
            inputs_CT = val_data['CT'].cuda()
            # inputs = torch.cat((inputs_RD, inputs_CT), dim=1)
            # inputs = inputs_CT*inputs_CT*(1+inputs_RD)
            inputs = torch.cat((inputs_CT, inputs_RD), dim=1)
            labels = val_data['Label'].cuda()
            val_outputs = net(inputs)
            val_outputs=torch.sigmoid(val_outputs)
            # value = torch.eq(val_outputs.argmax(dim=1), labels)
            # metric_count += len(value)
            # num_correct += value.sum().item()
            # pre_list += val_outputs.argmax(dim=1).tolist()
            val_outputs = torch.squeeze(val_outputs)
            labels = torch.squeeze(labels)
            try:
                pre_list += val_outputs.tolist()
                # value = torch.eq(val_outputs.argmax(dim=1), labels)
                # pre_list += val_outputs.argmax(dim=1).tolist()
                label_list += labels.tolist()
            except:
                pre_list.append(val_outputs.item())
                label_list.append(labels.item())
                # if abs(labels.item()-val_outputs.item())>0.5:
                #     print("Pre",val_outputs.item())
                #     print("Label", labels.item())
                #     print("Patient",val_data['Patient_name'])
        # try:
        dict_pre = cal_boot_mertices(pre_list, label_list)

        print("95 AUC",dict_pre['auc'])

        print("95 Acc", dict_pre['acc'])

        print("95 sen", dict_pre['sen'])

        print("95 spe",dict_pre['spe'])

        df_all = pd.DataFrame([dict_pre])
        df_all.to_csv(out_path+'scores24.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters", type=str, default='Img_classification_USA.json', help="parameters json")
    args = parser.parse_args()

    para_path = './parameters/' + args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)