import json
import os
import random
import numpy as np
import pandas as pd


def find_end(file_name):
    names = file_name.split("_")
    names_t = names[-1].split(".")
    return names_t[0]


def files_name(file_dir,df):
    data_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_name = os.path.join(root,file)
            file_name  = file_name.replace("\\","/")
            dict_temp = {}
            if 'CT' in file_name:
                img_name = file_name

                patient_name = img_name.replace('.nii.gz','')
                patient_name = patient_name.split('/')
                patient_name = patient_name[-1]
                if '_' in patient_name:
                    patient_name=patient_name.replace('_',' ')
                new_CT_name = file_name.replace(' ','_')
                os.rename(file_name,new_CT_name)

                df_s=df[df['patient'] ==patient_name]
                if df_s['Label'].item()>=2:
                    dict_temp['Label']=1
                else:
                    dict_temp['Label'] = 0
                mask_o_name = file_name.replace('CT','RS')
                RD_o_name = file_name.replace('CT', 'RD')
                dict_temp['CT']=img_name
                dict_temp['RS'] = mask_o_name
                dict_temp['RD'] = RD_o_name
                data_list.append(dict_temp)


    return data_list



if __name__ == '__main__' :
    dicts =  {
        "name": "RD",
        "description": "Lung CT RD RS",
        "reference": "",
        "licence":"",
        "release":"",
        "tensorImageSize": "4D",
        "modality": {
             "img": "image",
         },
         "labels": {
             "0": "Background",
             "1": "Lung",
             "2": "GTV",
         } }


    n_training_rate= 0.8
    n_val_rate =0.1
    n_test_rate=0.1
    file_dir_training = 'D:/RD_NRRD/Croped/'
    csv18_path = 'RD_features_2018_183.csv'
    csv19_path = 'RD_features_2019_166.csv'
    df18 = pd.read_csv(csv18_path)
    df19 = pd.read_csv(csv19_path)
    df = pd.concat([df18,df19])
    train_names = files_name(file_dir_training,df)
    random.shuffle(train_names)
    n = len(train_names)
    print("n=",n)
    train_set = train_names[:int(n*n_training_rate)]
    test_set = train_names[int(n*n_training_rate):int(n*(n_training_rate+n_val_rate))]
    val_set = train_names[int(1-n_test_rate):]
    dicts_train = {"numTraining": len(train_set), "numTest":len(test_set), "training": train_set, "test": test_set, "validation": val_set}

    dicts.update(dicts_train)
    with open("dataset_Croped_nii.json","w") as dump_f:
        json.dump(dicts,dump_f,indent=4)