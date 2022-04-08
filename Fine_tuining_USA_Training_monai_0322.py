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
    train_ds,val_ds=read_data_ds(data_path,section='test',train_section='training')
    # _,test_ds = read_data_ds(data_path,section='training')


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
    for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
        print(layer, param)


        print(net)

    net = frozen_Resnet(net)


    net.to(device)




    # loss_function = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1/270,1/70])).float()).to(device)
    # ww=torch.tensor([0.6]).cuda()
    # loss_function = torch.nn.BCELoss()
    loss_function = torch.nn.BCELoss()
    # loss_function = BCEFocalLoss(reduction='sum')
    # loss_function = FocalTverskyLoss()
    # loss_function = FocalLoss()
    # optimizer = torch.optim.Adam(net.parameters(), learning_rate,weight_decay=0.01)
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)#冻结模型参数
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)#余弦退火学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    accumulation_steps=1
    best_loss=1
    metric_values = []

    max_epochs = num_epoch
    best_epoch=0
    loss_s=0
    dict_all_list =[]
    fc_wight_list=[]
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
            # inputs = [inputs_CT]
            inputs = torch.cat((inputs_CT, inputs_RD), dim=1)
            # inputs = inputs_CT*(1+inputs_RD)
            #=============================检查输入
            # # #
            # ct_ = inputs[0,0].cpu().detach().numpy()
            # rd_ = inputs[0,1].cpu().detach().numpy()
            # # rs_ = data['RS'].cpu().detach().numpy()
            # # ct_ = abs(ct_-1)
            # plt.subplot(1, 3, 1)
            # plt.imshow(ct_[ :, :,32])
            # plt.subplot(1, 3, 2)
            # plt.imshow(rd_[ :, :,32])
            # # plt.subplot(1, 3, 3)
            # # plt.imshow(rs_[0,0,:, :,32])
            # plt.show()
            # CT_image = sitk.GetImageFromArray(ct_)
            # sitk.WriteImage(CT_image,'./CT.nii.gz')
            # rd_image = sitk.GetImageFromArray(rd_)
            # sitk.WriteImage(rd_image, './RD.nii.gz')
            # RS_image = sitk.GetImageFromArray(rs_)
            # sitk.WriteImage(RS_image, './RS.nii.gz')
            # print("Saving.....")
            #===========================================
            labels = data['Label'].cuda()
            labels = torch.unsqueeze(labels,1)
            labels = labels.to(torch.float32)

            # labels = torch.nn.functional.one_hot(labels, 2)


            outputs = net(inputs)

            # outputs = outputs.cpu()
            outputs=torch.sigmoid(outputs)
            # outputs = torch.softmax(outputs,dim=1)
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
            # loss.backward()  # 计算梯度
            #
            # optimizer.step()  # 反向传播，更新网络参数
            # optimizer.zero_grad()  # 清空梯度
            # optimizer.step()

            epoch_loss += loss.item()
            # epoch_len = len(train_ds) // train_loader.batch_size

            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
            try:
                pre_list += outputs.tolist()
            # value = torch.eq(val_outputs.argmax(dim=1), labels)
            # pre_list += val_outputs.argmax(dim=1).tolist()
                label_list += labels.tolist()
            except:
                pre_list.append(outputs.item())
                label_list.append(labels.item())

            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")



            loss_list.append(loss.item())
            scheduler.step()
            # for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
            #     if 'classififcation.5.weight' == layer:
            #         fc_wight_list.append(param)
            # if len(fc_wight_list) > 1:
            #     c = fc_wight_list[len(fc_wight_list) - 1] - fc_wight_list[len(fc_wight_list) - 2]
            #     summ = torch.sum(c)
            #     print("FC 差值", summ)
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            # train_bar.set_description("loss: %s" % (loss.item()))

        # acc,sen,spe=calculate_metric(label_list, pre_list)
        # AUC = metrics.roc_auc_score(label_list, pre_list)
        # print("Train AUC",AUC)
        score_dict=bootstrap_95metrics(label_list,pre_list)
        # dict_all_list.append(score_dict)
        #
        # df_temp = pd.DataFrame(dict_all_list)
        # df_temp.to_csv(out_path+'Scores.csv', index=False)
        AUC=score_dict['auc'][0]
        print("AUC", score_dict['auc'])
        print("acc", score_dict['acc'])
        print("sen", score_dict['sen'])
        print("spe", score_dict['spe'])
        pre_list = torch.tensor(pre_list)
        pre_list = pre_list.to(torch.float32).cuda()
        label_list = torch.tensor(label_list)
        label_list = label_list.to(torch.float32).cuda()
        evl_loss = loss_function(pre_list, label_list)
        print("Train AVG Loss", evl_loss.item())
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        # print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
            print(layer, param)
            sum = torch.sum(abs(param))
            print("A,B:",(abs(param[0])/sum).item(),(abs(param[1]/sum)).item())
            break



        if (epoch + 1) % val_interval == 0:


                print("val==================")
                pre_list = []
                label_list = []
                with torch.no_grad():
                    net.eval()
                    num_correct = 0.0
                    metric_count = 0
                    for val_data in val_loader:
                        inputs_RD = val_data['RD'].cuda()
                        inputs_CT = val_data['CT'].cuda()
                        # inputs = [inputs_CT]
                        inputs = torch.cat((inputs_CT, inputs_RD), dim=1)
                        # inputs = inputs_CT*(1+inputs_RD)

                        labels = val_data['Label'].cuda()
                        labels = torch.unsqueeze(labels, 1)
                        labels = labels.to(torch.float32)
                        # labels = torch.nn.functional.one_hot(labels, 2)

                        outputs = net(inputs)

                        # outputs = outputs.cpu()
                        outputs = torch.sigmoid(outputs)
                        outputs = torch.squeeze(outputs)
                        labels = torch.squeeze(labels)
                        try:
                            pre_list += outputs.tolist()
                            label_list += labels.tolist()
                        except:
                            pre_list.append(outputs.item())
                            label_list.append(labels.item())
                    # acc,sen,spe=calculate_metric(label_list, pre_list)
                    # AUC = metrics.roc_auc_score(label_list, pre_list)
                    # pre_dict = []
                    # for pp in range(len(label_list)):
                    #     dict_t = {}
                    #     dict_t[label_list[pp]] = pre_list[pp]
                    #     pre_dict.append(dict_t)
                    # print("Label:pre",pre_dict)
                    # print("val AUC", AUC)
                    score_dict = bootstrap_95metrics(label_list, pre_list)
                    # print("Label:pre",pre_dict)
                    sen=score_dict['sen'][0]
                    spe=score_dict['spe'][0]
                    AUC=np.round(score_dict['auc'][0],2)
                    print("val AUC", score_dict['auc'])
                    print("val acc", score_dict['acc'])
                    print("val sen", score_dict['sen'])
                    print("val spe", score_dict['spe'])
                    dict_all_list.append(score_dict)
                    #
                    df_temp = pd.DataFrame(dict_all_list)

                    # df_temp.to_csv(out_path +'Scores.csv', index=False)
                    dict_cal = {}
                    dict_cal['pre']=pre_list
                    dict_cal['label']=label_list
                    dict_cal['auc']=np.ones(len(pre_list))*AUC
                    df_cal = pd.DataFrame(dict_cal)
                    df_cal=df_cal
                    df_cal.to_csv(out_path + str(epoch)+'_Scores.csv', index=False)
                    pre_list = torch.tensor(pre_list)
                    pre_list = pre_list.to(torch.float32).cuda()
                    label_list = torch.tensor(label_list)
                    label_list = label_list.to(torch.float32).cuda()
                    evl_loss = loss_function(pre_list, label_list)
                    evl_loss = evl_loss.item()
                    print("EVL Loss",evl_loss)
                    score = AUC+sen+0.5*spe
                    # if evl_loss<best_loss:
                    # for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
                    #     if 'classififcation.5.weight' == layer:
                    #         fc_wight_list.append(param)
                    # if len(fc_wight_list) > 1:
                    #     c = fc_wight_list[len(fc_wight_list) - 1] - fc_wight_list[len(fc_wight_list) - 2]
                    #     summ = torch.sum(c)
                    #     print("FC 差值", summ)
                    if score > best_metric:
                        # best_loss=evl_loss
                        best_metric=score
                        best_auc=AUC

                        print("best score",score)
                        best_metric_epoch = epoch + 1
                        min_epoch_loss= epoch_loss
                        # if not os.path.exists(load_path):
                        #     os.makedirs(load_path)
                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        torch.save(net.state_dict(), load_path + 'bestmodel24.pth')
                        # torch.save(net.state_dict(), out_path + 'bestmodel.pth')
                        print("saved new best metric model")
                        print(
                            "current epoch: {} current accuracy: {:.4f} "
                            "best AUC: {:.4f} at epoch {}".format(
                                epoch + 1, AUC, best_auc, best_metric_epoch
                            )
                        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters", type=str, default='Img_classification_USA.json', help="parameters json")
    args = parser.parse_args()

    para_path = './parameters/' + args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)