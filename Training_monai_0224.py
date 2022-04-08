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
from LOSS.focal_loss import *
from LOSS.Focal_Tversky_loss import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    train_ds,val_ds=read_data_ds(data_path,section='validation',train_section='training')
    _,test_ds = read_data_ds(data_path,section='inde_test')

    # 读取数据
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2,pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, drop_last=True, num_workers=2)
    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    out_path = './training_results/' + structure + '/' + time_save + '/'




    load_path = './training_results/' + structure + '/'


    net = load_model(structure)

    print(net)





    if continue_training == True:
        print('loading model............')
        # try:
        net.load_state_dict(torch.load(load_path + 'bestmodel.pth'))

        for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
            print(layer, param)


        print(net)


    # net = frozen_Resnet(net)


    net.to(device)





    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    # start a typical PyTorch training
    val_interval = 1

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    accumulation_steps=1
    writer = SummaryWriter()
    max_epochs = num_epoch
    loss_s=0
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        net.train()
        epoch_loss = 0
        step = 0
        train_bar = tqdm(train_loader)
        pre_list=[]
        label_list=[]
        loss_list=[]
        for i, data in enumerate(train_bar):
            step += 1

            inputs_RD = data['RD'].cuda()
            inputs_CT = data['CT'].cuda()

            inputs = torch.cat((inputs_CT, inputs_RD), dim=1)


            labels = data['Label'].cuda()
            labels = torch.unsqueeze(labels,1)
            labels = labels.to(torch.float32)



            outputs = net(inputs)
            outputs=torch.sigmoid(outputs)
            loss = loss_function(outputs, labels)
            loss = loss / accumulation_steps
            loss_s += loss.item()
            loss.backward()
            epoch_len = len(train_ds) // train_loader.batch_size
            if ((i + 1) % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar("train_loss", loss_s, epoch_len * epoch + step)
                train_bar.set_description("loss: %s" % (loss_s))
                loss_s=0

            epoch_loss += loss.item()

            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
            try:
                pre_list += outputs.tolist()
                label_list += labels.tolist()
            except:
                pre_list.append(outputs.item())
                label_list.append(labels.item())




            loss_list.append(loss.item())
            scheduler.step()


        acc,sen,spe=calculate_metric(label_list, pre_list)
        AUC = metrics.roc_auc_score(label_list, pre_list)
        print("Train AUC",AUC)

        pre_list = torch.tensor(pre_list)
        pre_list = pre_list.to(torch.float32).cuda()
        label_list = torch.tensor(label_list)
        label_list = label_list.to(torch.float32).cuda()
        evl_loss = loss_function(pre_list, label_list)
        print("Train AVG Loss", evl_loss.item())
        writer.add_scalar("train_AUC", AUC, epoch)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        if (epoch + 1) % val_interval == 0:


                print("val==================")
                pre_list = []
                label_list = []
                with torch.no_grad():
                    net.eval()

                    for val_data in val_loader:
                        inputs_RD = val_data['RD'].cuda()
                        inputs_CT = val_data['CT'].cuda()

                        inputs = torch.cat((inputs_CT, inputs_RD), dim=1)


                        labels = val_data['Label'].cuda()
                        labels = torch.unsqueeze(labels, 1)
                        labels = labels.to(torch.float32)


                        outputs = net(inputs)

                        outputs = torch.sigmoid(outputs)
                        outputs = torch.squeeze(outputs)
                        labels = torch.squeeze(labels)
                        try:
                            pre_list += outputs.tolist()
                            label_list += labels.tolist()
                        except:
                            pre_list.append(outputs.item())
                            label_list.append(labels.item())
                    acc,sen,spe=calculate_metric(label_list, pre_list)
                    AUC = metrics.roc_auc_score(label_list, pre_list)
                    pre_dict = []
                    for pp in range(len(label_list)):
                        dict_t = {}
                        dict_t[label_list[pp]] = pre_list[pp]
                        pre_dict.append(dict_t)
                    print("Label:pre",pre_dict)
                    print("val AUC", AUC)

                    pre_list = torch.tensor(pre_list)
                    pre_list = pre_list.to(torch.float32).cuda()
                    label_list = torch.tensor(label_list)
                    label_list = label_list.to(torch.float32).cuda()
                    evl_loss = loss_function(pre_list, label_list)
                    evl_loss = evl_loss.item()
                    print("EVL Loss",evl_loss)
                    score = AUC+sen+0.5*spe

                    if score > best_metric:

                        print("Best score",score)
                        best_metric=score
                        best_auc=AUC

                        best_metric_epoch = epoch + 1
                        if not os.path.exists(load_path):
                            os.makedirs(load_path)
                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        torch.save(net.state_dict(), load_path + 'bestmodel.pth')
                        torch.save(net.state_dict(), out_path + 'bestmodel.pth')
                        print("saved new best metric model")
                    print(
                        "current epoch: {} current accuracy: {:.4f} "
                        "best AUC: {:.4f} at epoch {}".format(
                            epoch + 1, AUC, best_auc, best_metric_epoch
                        )
                    )


        if (epoch + 1) % 2== 0:
            print("Test===============================")
            net.eval()
            with torch.no_grad():
                pre_list =[]
                label_list=[]
                for val_data in test_loader:
                    inputs_RD = val_data['RD'].cuda()
                    inputs_CT = val_data['CT'].cuda()

                    inputs = torch.cat((inputs_CT, inputs_RD), dim=1)
                    labels = val_data['Label']

                    val_outputs = net(inputs)

                    val_outputs=torch.sigmoid(val_outputs)

                    labels = torch.squeeze(labels)
                    pre_list.append(val_outputs.item())
                    label_list.append(labels.item())
                pre_dict = []
                for pp in range(len(label_list)):
                    dict_t={}
                    dict_t[label_list[pp]] = pre_list[pp]
                    pre_dict.append(dict_t)
                print("Label:pre", pre_dict)


                acc,sen,spe=calculate_metric(label_list, pre_list)
                AUC = metrics.roc_auc_score(label_list, pre_list)
                print("Test AUC====",AUC)
                pre_list = torch.tensor(pre_list)
                pre_list = pre_list.to(torch.float32).cuda()
                label_list = torch.tensor(label_list)
                label_list = label_list.to(torch.float32).cuda()
                evl_loss = loss_function(pre_list, label_list)
                print("test Loss", evl_loss.item())



    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters", type=str, default='Img_classification.json', help="parameters json")
    args = parser.parse_args()

    para_path = './parameters/' + args.parameters

    with open(para_path, 'r') as f:  
        data_para = json.load(f)
    train(data_para)