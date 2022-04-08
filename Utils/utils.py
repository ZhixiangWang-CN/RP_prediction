import pydicom
import numpy as np
import os
import SimpleITK as sitk
import torch
from monai.data import CacheDataset
import matplotlib.pyplot as plt
import sys
from monai.transforms import *
from skimage import transform
import math
import skimage.morphology as sm
import numpy as np
import nrrd
import torch.nn as nn
from monai.config import *

import pandas as pd
from sklearn.metrics import *
from typing import *
import random
from Utils.BraTS_datalist import *
from skimage import measure
from matplotlib.cm import get_cmap
import os
import re
import sklearn.metrics as metrics
from pandas import *
from model.load_model import *
def de_interpolate(raw_tensor,index=2):
    """
    F.interpolate(source, scale_factor=scale, mode="nearest")的逆操作！
    :param raw_tensor: [B, C, H, W]
    :return: [B, C, H // 2, W // 2]
    """
    out = raw_tensor[0::index, 0::index, 0::index]
    return out
def plot_3D_list(img_list,save_img_name,down_size):
    cam_3d=img_list[0]

    # cam_3d = torch.tensor(cam_3d)
    cam_3d = de_interpolate(cam_3d,index=2)
    # cam_3d = cam_3d.numpy()

    ax = plt.subplot(111, projection='3d')
    # # Combine both previous iterations
    filled = np.ones(cam_3d.shape, dtype=np.bool)
    filled[cam_3d == 0] = False
    # colors = np.empty((*cam_3d.shape, 4))
    # colors[..., -1] = 0.8

    ax.voxels(filled, facecolors='#292421',alpha=0.2)
    # plt.show()
    img_t_list = torch.tensor(img_list[1])
    # cam_3d,_=torch.max(img_t_list,dim=0)
    cam_3d= torch.tensor(img_list[1])
    cam_3d=cam_3d.numpy()
    cam_3d = de_interpolate(cam_3d)
    cam_3d[cam_3d < 0.6] = 0
    filled = np.ones(cam_3d.shape, dtype=np.bool)
    filled[cam_3d == 0] = False
    colors = np.empty((*cam_3d.shape, 4))
    colors[..., -1] = 0.5
    cmap = get_cmap('rainbow')
    colors[..., :3] = cmap(cam_3d)[..., :-1]
    ax.voxels(filled, facecolors=colors)
    plt.savefig(save_img_name)
    # plt.show()

# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def plot_auc(y_test,y_pred,save_name):
    print("auc:")
    fpr, tpr, thread = metrics.roc_curve(np.array(y_test), np.array(y_pred))
    x=metrics.auc(fpr, tpr)
    print(x)
    plt.title("ROC curve of %s (AUC = %.4f)" % ('lightgbm', x))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr,tpr)  # use pylab to plot x and y
    plt.savefig(save_name)
    csv_save_name = save_name.replace("tif",'csv')
    data={'Pre':y_pred,'Label':y_test}
    df = DataFrame(data)
    df.to_csv(csv_save_name,index=False)
    plt.clf()


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        print("Weighted",classname)
        nn.init.normal_(m.weight,0,0.01)
        nn.init.constant_(m.bias, 0.0)

def weight_net(model):
    model.apply(weights_init)
    return model
def frozen_Resnet(model):
    for k, v in model.named_parameters():
        if 'resnet' in k :
                v.requires_grad = False  # 固定参数

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Can train:",name)
    return model
def frozen_Body(model):
    # res_names=[]
    # for name in model.state_dict():
    #     if 'resnet' in name:
    #         res_names.append(name)
    for k, v in model.named_parameters():
        if 'resnet' in k or 'classification' in k :
            # if 'layer4.2'not in k:
                v.requires_grad = False  # 固定参数

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Can train:",name)
    return model
def find_all_files(path):
    files_list =[]
    for root,dirs,files in os.walk(path):
        for file in files:
            name = os.path.join(root, file)
            name= name.replace('\\','/')
            files_list.append(name)
    return files_list
def Sep_CT_RD_RS(files_list):
    CT_List = []
    RD_list = []
    RS_list = []
    for name in files_list:
        file_name = name.split('/')
        end_name = file_name[-1]
        if (re.search('CT.', end_name, re.IGNORECASE)or re.search('image', end_name, re.IGNORECASE))and '.dcm'in name:
            CT_List.append(name)
        elif re.search('RS.', end_name, re.IGNORECASE)and '.dcm'in name:
            RS_list.append(name)

        elif re.search('rd.', end_name, re.IGNORECASE)and '.dcm'in name:

            RD_list.append(name)
    return CT_List,RD_list,RS_list



def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


class Expand_dim(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            img = d[k]

            img = np.expand_dims(img,axis=0)
            d[k]=img
        return d
class Compress_dim(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            img = d[k][0]

            d[k]=img
        return d
def convert_clinical_features(clinical_data):
    clinical_numpy = []
    for o in clinical_data:
        temp = o
        o_ = o.split(',')
        numbers = list(map(float, o_))
        clinical_numpy.append(numbers)
    clinical_numpy = np.array(clinical_numpy)
    return clinical_numpy
class Crop_by_RS(MapTransform):
    def __init__(self):
        self.keys =['RS']
    def __call__(self, data):
        d = dict(data)
        mask = data['RS']
        RD = data['RD']



        RD_mask_pos = np.where((RD>=30) & (mask==1))
        shape_RD_mask = [max(RD_mask_pos[0]) , min(RD_mask_pos[0]), max(RD_mask_pos[1]) , min(RD_mask_pos[1]),
                    max(RD_mask_pos[2]) , min(RD_mask_pos[2])]
        CT = data['CT']
        CT_roi = CT[shape_RD_mask[1]:shape_RD_mask[0],shape_RD_mask[3]:shape_RD_mask[2],shape_RD_mask[5]:shape_RD_mask[4]]
        RD_roi = RD[shape_RD_mask[1]:shape_RD_mask[0], shape_RD_mask[3]:shape_RD_mask[2],
                 shape_RD_mask[5]:shape_RD_mask[4]]
        Mask_roi = mask[shape_RD_mask[1]:shape_RD_mask[0], shape_RD_mask[3]:shape_RD_mask[2],
                 shape_RD_mask[5]:shape_RD_mask[4]]
        d['CT'] = CT_roi
        d['RD'] = RD_roi
        d['RS'] =Mask_roi
        return d


class Norm_CT_RD(MapTransform):
    def __init__(self):
        self.keys =['RS']
    def __call__(self, data):
        d = dict(data)

        RD = data['RD']
        RD[RD>80]=80
        RD = RD / 80
        d['RD'] = RD
        CT = data['CT']
        CT[CT==0]=-1000
        CT[CT > 100] = 100
        CT[CT < -1000] = -1000
        CT=(CT+1000)/1100
        d['CT'] = CT
        return d

# code resourse: https://www.atyun.com/23342.html
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)
class RandomAug(MapTransform):
    def __init__(self):
        self.keys =0
    def __call__(self, data):
        d = dict(data)

        images = data['image']
        targets= np.zeros(images.shape)
        randx= random.randint(0,10)
        randy = random.randint(0,10)
        randz = random.randint(0,10)
        f = random.randint(0,1)
        if f>0.5:
            targets[:,randx:,randy:,randz:]=images[:,:84-randx,:84-randy,:84-randz]
        else:
            targets[:, :84-randx,:84-randy,:84-randz] = images[:,randx:,randy:,randz:]
        data['image']=targets
        return d

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
    spe = TN / float(TN+FP)
    print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    print('Sensitivity:',TP / float(TP+FN))
    print('Specificity:',TN / float(TN+FP))
    return acc,sen,spe
class Renorm_RD(MapTransform):
    def __init__(self):
        self.keys =['RS']
    def __call__(self, data):
        d = dict(data)

        RD = data['RD']
        RD = RD*1.25
        d['RD'] = RD

        return d
class ConvertToLungMask(MapTransform):

    def __call__(self, data):
        d = dict(data)
        mask = d[self.keys]
        # mask[mask==1]=0
        # mask[mask==4]=0
        # mask[mask==3]=2
        # mask[mask==2]=1
        mask[mask==3]=2
        mask[mask!=2]=0
        mask[mask==2]=1
        d[self.keys] = mask
        #输出为：Tumor core,Whole tumor,Enhance Tumor Core[TC,WT,ET]
        return d

class BraTSDataset(Randomizable, CacheDataset):

    resource = {

    }
    md5 = {

    }

    def __init__(
            self,

            json_path: str,
            section: str,
            transform: Union[Sequence[Callable], Callable] = (),
            seed: int = 0,
            val_frac: float = 0.2,
            cache_num: int = sys.maxsize,
            cache_rate: float = 1.0,
            num_workers: int = 0,
    ) -> None:

        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)

        self.indices: np.ndarray = np.array([])
        data = self._generate_data_list(json_path)
        # as `release` key has typo in Task04 config file, ignore it.
        property_keys = [
            "description",
            "reference",
            "licence",
            "tensorImageSize",
            "modality",
            "numTraining",
            "numTest",
        ]
        self._properties = load_BraTS_datalist(json_path, property_keys)
        if transform == ():
            transform = LoadImaged(["t1", "t2", "t1ce", "flair", "seg"])
        super().__init__(data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.

        """
        return self.indices

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def get_properties(self, keys: Optional[Union[Sequence[str], str]] = None):
        """
        Get the loaded properties of dataset with specified keys.
        If no keys specified, return all the loaded properties.

        """
        if keys is None:
            return self._properties
        elif self._properties is not None:
            return {key: self._properties[key] for key in ensure_tuple(keys)}
        else:
            return {}

    def _generate_data_list(self, json_path: str) -> List[Dict]:
        # section = "training" if self.section in ["training", "validation"] else "test"
        section = self.section
        datalist = load_BraTS_datalist(json_path, True, section)
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        # if self.section == "test":
        return datalist
        # length = len(datalist)
        # indices = np.arange(length)
        # self.randomize(indices)
        #
        # val_length = int(length * self.val_frac)
        # if self.section == "training":
        #     self.indices = indices[val_length:]
        # else:
        #     self.indices = indices[:val_length]
        #
        # return [datalist[i] for i in self.indices]

class LoadImaged_zx(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadImage`,
    must load image and metadata together. If loading a list of files in one key,
    stack them together and add a new dimension as the first dimension, and use the
    meta data of the first image to represent the stacked result. Note that the affine
    transform of all the stacked images should be same. The output metadata field will
    be created as ``key_{meta_key_postfix}``.

    It can automatically choose readers based on the supported suffixes and in below order:
    - User specified reader at runtime when call this loader.
    - Registered readers from the latest to the first in list.
    - Default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
    (npz, npy -> NumpyReader), (others -> ITKReader).

    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
        image_only: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            reader: register reader to load image file and meta data, if None, still can register readers
                at runtime or use the default readers. If a string of reader name provided, will construct
                a reader object with the `*args` and `**kwargs` parameters, supported reader name: "NibabelReader",
                "PILReader", "ITKReader", "NumpyReader".
            dtype: if not None convert the loaded image data to this data type.
            meta_key_postfix: use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The meta data is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting: whether allow to overwrite existing meta data of same key.
                default is False, which will raise exception if encountering existing key.
            image_only: if True return dictionary containing just only the image volumes, otherwise return
                dictionary containing image data array and header dict per input key.
            allow_missing_keys: don't raise exception if key is missing.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.
        """
        super().__init__(keys, allow_missing_keys)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix
        self.overwriting = overwriting


    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key in self.key_iterator(d):
            data,head = nrrd.read(d[key])

            d[key] = data

        return d