import numpy as np
import torch
from monai.apps import DecathlonDataset
from Utils.utils import BraTSDataset
from Utils.utils import *
from monai.transforms import *

from monai.data import DataLoader
import matplotlib.pyplot as plt
import SimpleITK as sitk

def read_data_ds(data_path='../Tools/dataset_Croped_nii_0127.json',spatial_size=[230,180,256],cache_num=2,train_section='training',section='validation'):

    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["CT", "RD"]),

            Norm_CT_RD(),
            Expand_dim(keys=["CT", "RD"]),

            # RandSpatialCropd(
            #     keys=["CT", "RS", "RD"], roi_size=spatial_size, random_size=False,random_center=False
            # ),

            RandRotateD(keys=['CT', 'RD',], prob=0.5),
            RandFlipd(keys=["CT", "RD"], prob=0.5,spatial_axis=[0,1,2]),
            RandAffineD(keys=["CT", "RD"], prob=0.5),
            # RandZoomd(keys=["CT", "RS", "RD"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
            RandSpatialCropd(
                keys=["CT", "RS", "RD"], roi_size=[70,70,70],max_roi_size=[84,84,84], random_size=True,random_center=False
            ),
            SpatialPadD(keys=["CT", "RS", "RD"], spatial_size=[84,84,84]),

            ToTensord(keys= ['CT',"RS","RD"]),
        ]
    )
    val_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["CT", "RD"]),
            Norm_CT_RD(),
            Expand_dim(keys=["CT",  "RD"]),
            ResizeD(keys=["CT", "RS", "RD"], spatial_size=[84, 84, 84]),
            ToTensord(keys=['CT',  "RD"]),
        ]
    )
    train_ds = BraTSDataset(
        json_path=data_path,
        transform=train_transform,
        section=train_section,
        num_workers=0,
        cache_num=cache_num,


    )
    val_ds = BraTSDataset(
        json_path=data_path,
        transform=val_transform,
        section=section,
        num_workers=0,
        cache_num=cache_num

    )

    return train_ds,val_ds


if __name__ == '__main__':
    # path = 'C:/Softwares/Codes/BrainMRI/BrainMRISegmentation/utils/shuffle_UNet.json'
    train_ds,val_ds=read_data_ds()
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    i=0

    for epoch in range(3):
        for batch_data in train_loader:
            inputs_RD = batch_data['RD'].cuda()
            inputs_CT = batch_data['CT'].cuda()
            ct_ = inputs_CT.cpu().detach().numpy()
            rd_ = inputs_RD.cpu().detach().numpy()
            rs_ = batch_data['RS'].cpu().detach().numpy()
            plt.subplot(1, 3, 1)
            plt.imshow(ct_[0, 0, :, :, 128])
            plt.subplot(1, 3, 2)
            plt.imshow(rd_[0, 0, :, :, 128])
            plt.subplot(1, 3, 3)
            plt.imshow(rs_[0, 0, :, :, 128])
            plt.show()
            CT_image = sitk.GetImageFromArray(ct_[0,0])
            sitk.WriteImage(CT_image, './CT.nii.gz')
            rd_image = sitk.GetImageFromArray(rd_[0,0])
            sitk.WriteImage(rd_image, './RD.nii.gz')
            RS_image = sitk.GetImageFromArray(rs_[0,0])
            sitk.WriteImage(RS_image, './RS.nii.gz')
            print("Saving.....")

