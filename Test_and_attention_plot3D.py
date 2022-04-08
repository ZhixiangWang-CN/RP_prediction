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
from skimage.transform import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn import metrics
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, ImageDataset
# from Loader.loader_CT_test import read_data_ds as read_data_ds_test
from gcam import gcam

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
    train_ds,val_ds=read_data_ds(data_path)
    _,test_ds = read_data_ds(data_path,section='test')
    # 读取数据
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=True, num_workers=2)
    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    out_path = './test_results/' + structure + '/' + time_save + '/'

    # pic_save_path = out_path + 'images/'


    load_path = './training_results/' + structure + '/'

    # 定义判别器  #####Discriminator######使用多层网络来作为判别器

    net = load_model(structure)

    print(net)




    save_num=0
    best_recall=0
    auc_list = []
    min_epoch_loss= 99

    print('loading model............')

    net.load_state_dict(torch.load(load_path + 'bestmodel.pth'))

    # for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
    #     print(layer, param)
    checkpoint = torch.load(load_path + 'bestmodel.pth')
    for k, v in checkpoint.items():
        print(k)
    print(net)
    Method='gcam'
    layer = 'resnet.layer4'

    # net = gcam.inject(net_, output_dir=out_path + layer, backend=Method, layer=layer, label=1, save_maps=True)
    net.to(device)
    net = gcam.inject(net, output_dir=out_path, backend=Method, layer=layer, label=0, save_maps=True)
    # loss_function = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1/270,1/70])).float()).to(device)
    loss_function = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(net.parameters(), learning_rate,weight_decay=0.01)
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    max_epochs = num_epoch

    print("Test===============================")

    i=0
    for val_data in train_loader:
        inputs_RD = val_data['RD'].cuda()
        inputs_CT = val_data['CT'].cuda()

        inputs = torch.cat((inputs_CT, inputs_RD), dim=1)
        labels = val_data['Label'].cuda()
        val_outputs = net(inputs)
        # print(i)
        label = labels[0].item()
        patient_name = val_data['Patient_name'][0]
        loader_path =out_path+layer+'/attention_map_'+str(i)+'_0_0.nii.gz'
        IMG_map = sitk.ReadImage(loader_path, sitk.sitkFloat32)
        IMG_map = sitk.GetArrayFromImage(IMG_map)
        image_resized = resize(IMG_map, (84, 84, 84))
        ##注意这行只能对ggcam使用

        # vals, counts = np.unique(image_resized, return_counts=True)
        # index = np.argmax(counts)
        # minn = image_resized.min()
        # image_resized[image_resized==index]=minn


        image_resized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())
        inputs_CT=inputs_CT[0,0].cpu().detach().numpy()
        inputs_RD = inputs_RD[0, 0].cpu().detach().numpy()
        # outwithRD = outwithRD[0].cpu().detach().numpy()
        img_save_name = out_path+patient_name+'.png'

        Image_list = [inputs_CT,image_resized]
        down_size=4
        # plot_3D_list(
        #     Image_list,img_save_name,down_size
        # )
        img_save_name = out_path + patient_name
        CT_IMG = sitk.GetImageFromArray(inputs_CT)
        sitk.WriteImage(CT_IMG,img_save_name+'_'+str(label)+'CT.nii.gz')
        RD_IMG = sitk.GetImageFromArray(inputs_RD)
        sitk.WriteImage(RD_IMG, img_save_name+'_'+str(label) + 'RD.nii.gz')
        MAP = sitk.GetImageFromArray(image_resized)
        sitk.WriteImage(MAP, img_save_name +'_'+str(label)+ 'MAP.nii.gz')
        print("SAVING...",img_save_name)
    # AUC = metrics.roc_auc_score(label_list, pre_list)
    # print("Test AUC====",AUC)
    # writer.add_scalar("test_AUC", AUC, epoch + 1)
        i+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters", type=str, default='Img_classification.json', help="parameters json")
    args = parser.parse_args()

    para_path = './parameters/' + args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)