import pickle
import torch
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

def Area_nums():
    area_channel = 1
    return area_channel
def parse_id():


#    with open("traffic_shanghai.pkl", "rb") as f:
#        data = pickle.load(f)
#    observed_values=data[0:960]
#    
#    data = np.load('bs_shanghai.npz')
#    observed_values=data['bs_record'][:960,:170]

#    data = np.load('bs_beijing.npz')
#    observed_values=data['bs_record'][:4000,:170]
###    
#the name of your dataset
    data = np.load('netdata.npz')
    observed_value=data['traffic_features']
    observed_values = np.expand_dims(observed_value,axis = -1)
    #observed_values = np.load('netdata.npz')['TrafficFeatures']
    #observed_values = np.load('d1.npz')['packet_feature'].reshape(2500,1000,4)[:1672,:33,:]
    # min_val = np.min(observed_values, axis=-1)
    # max_val = np.max(observed_values, axis=-1)
    # normalized_data = (observed_values.T- min_val) / (max_val - min_val)
    # observed_values = normalized_data.T


    # observed_values = np.expand_dims(observed_values, axis=-1)
    # observed_values = np.stack(observed_values)

    # batch_size = 320
    # sequence_length = 1
    # signal_length = 120
    # period = 30
    #

    return observed_values


#获取数据集中文件标题-----即获得病人id号码
def get_idlist():
    data=np.load('netdata.npz')
    patient_id=len(data['traffic_features'])
    
    
    return patient_id


class Physio_Dataset(Dataset):
    def __init__(self, eval_length=10, use_index_list=None,seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

      
        self.observed_values = np.array(parse_id())

#            mean = np.mean(self.observed_values, axis=1, keepdims=True)
#            std_dev = np.std(self.observed_values, axis=1, keepdims=True)
#            self.observed_values = (self.observed_values - mean) / std_dev

    
                
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "timepoints": np.arange(self.eval_length),
            "idex_test": index,
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16):

    # only to obtain total length of dataset
    dataset = Physio_Dataset(seed=seed)
    indlist = np.arange(len(dataset))

#
    num_train = (int)(len(dataset) * 0.8)
    train_index = indlist[:num_train]
    test_index = indlist[num_train:]
    
    
#    num_train=1
#    train_index = indlist[:num_train]
#    test_index = train_index
#    test_index = [5]
#    test_index = [4943,4944,4945,4946,4936,4923,4926,4927,4680,4681,4682,4670,4671,4672,4673,4674]
#    test_index = [4686,4687,4688,4683,4660,4661,4662,4664,4564,4565,4572,4557,4569,4570,4743,4744]
#    test_index = [4701,4702,4703,4823,4824,4825,4820,4834,4835,4837,4838,4801,4802,4803,4771,4772]
#    test_index = [5070,5071,5072,5073,5074,5076,5077,5078,5079,5080,5081,5082,5086,5087,5088,5091]
    np.random.seed(seed)
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)

    #
    # num_train = (int)(len(dataset) * 0.8)
    # train_index = remain_index[:num_train]
    valid_index = test_index

    dataset = Physio_Dataset(
        use_index_list=train_index,seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Physio_Dataset(
        use_index_list=valid_index, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=1)
    test_dataset = Physio_Dataset(
        use_index_list=test_index, seed=seed
    )#test_index
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader

