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

    train_ds,val_ds=read_data_ds(data_path)
    _,test_ds = read_data_ds(data_path,section='test')

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, drop_last=True, num_workers=2)

    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    out_path = './test_results/' + structure + '/' + time_save + '/'




    load_path = './training_results/' + structure + '/'


    net = load_model(structure)

    print(net)






    print('loading model............')

    net.load_state_dict(torch.load(load_path + 'bestmodel.pth'))


    checkpoint = torch.load(load_path + 'bestmodel.pth')
    for k, v in checkpoint.items():
        print(k)
    print(net)
    Method='gcam'
    layer = 'resnet.layer4'

    # net = gcam.inject(net_, output_dir=out_path + layer, backend=Method, layer=layer, label=1, save_maps=True)
    net.to(device)
    net = gcam.inject(net, output_dir=out_path, backend=Method, layer=layer, label=0, save_maps=True)


    print("Test===============================")

    i=0
    for val_data in train_loader:
        inputs_RD = val_data['RD'].cuda()
        inputs_CT = val_data['CT'].cuda()

        inputs = torch.cat((inputs_CT, inputs_RD), dim=1)
        labels = val_data['Label'].cuda()
        val_outputs = net(inputs)

        label = labels[0].item()
        patient_name = val_data['Patient_name'][0]
        loader_path =out_path+layer+'/attention_map_'+str(i)+'_0_0.nii.gz'
        IMG_map = sitk.ReadImage(loader_path, sitk.sitkFloat32)
        IMG_map = sitk.GetArrayFromImage(IMG_map)
        image_resized = resize(IMG_map, (84, 84, 84))


        image_resized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())
        inputs_CT=inputs_CT[0,0].cpu().detach().numpy()
        inputs_RD = inputs_RD[0, 0].cpu().detach().numpy()

        img_save_name = out_path+patient_name+'.png'

        Image_list = [inputs_CT,image_resized]

        img_save_name = out_path + patient_name
        CT_IMG = sitk.GetImageFromArray(inputs_CT)
        sitk.WriteImage(CT_IMG,img_save_name+'_'+str(label)+'CT.nii.gz')
        RD_IMG = sitk.GetImageFromArray(inputs_RD)
        sitk.WriteImage(RD_IMG, img_save_name+'_'+str(label) + 'RD.nii.gz')
        MAP = sitk.GetImageFromArray(image_resized)
        sitk.WriteImage(MAP, img_save_name +'_'+str(label)+ 'MAP.nii.gz')
        print("SAVING...",img_save_name)
        i+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters", type=str, default='Img_classification.json', help="parameters json")
    args = parser.parse_args()

    para_path = './parameters/' + args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)