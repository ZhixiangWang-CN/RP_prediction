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
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn import metrics
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, ImageDataset
# from Loader.loader_CT_test import read_data_ds as read_data_ds_test


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
    best_score = 0
    train_ds,val_ds=read_data_ds(data_path,section='validation',train_section='training')



    # 读取数据
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2,pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, drop_last=True, num_workers=2)

    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    out_path = './training_results/' + structure + '/' + time_save + '/'




    load_path = './training_results/' + structure + '/'



    net = load_model(structure)

    print(net)





    print('loading model............')

    net.load_state_dict(torch.load(load_path + 'bestmodel.pth'))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for layer, param in net.state_dict().items():
        print(layer, param)


        print(net)

    net = frozen_Resnet(net)


    net.to(device)





    loss_function = torch.nn.BCELoss()


    optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    val_interval = 1
    best_metric = -1

    epoch_loss_values = []
    accumulation_steps=1


    max_epochs = num_epoch


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

            loss = loss / accumulation_steps  # 可选，如果损失要在训练样本上取平均
            loss_s += loss.item()
            loss.backward()  # 计算梯度
            epoch_len = len(train_ds) // train_loader.batch_size
            if ((i + 1) % accumulation_steps) == 0:
                optimizer.step()  # 反向传播，更新网络参数
                optimizer.zero_grad()  # 清空梯度
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


        pre_list = torch.tensor(pre_list)
        pre_list = pre_list.to(torch.float32).cuda()
        label_list = torch.tensor(label_list)
        label_list = label_list.to(torch.float32).cuda()
        evl_loss = loss_function(pre_list, label_list)
        print("Train AVG Loss", evl_loss.item())
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
            print(layer, param)
            sum = torch.sum(abs(param))
            print("A,B:",(abs(param[0])/sum).item(),(abs(param[1]/sum)).item())
            break

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        torch.save(net.state_dict(), load_path + 'bestmodel24.pth')
        print("Saved Model")

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
                    print("Test AUC", AUC)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters", type=str, default='Img_classification_USA.json', help="parameters json")
    args = parser.parse_args()

    para_path = './parameters/' + args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)