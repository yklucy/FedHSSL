import pickle
import os
import sys
import pandas as pd
import random
import logging
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
sys.path.append(os.getcwd())
from data.augmentation_img import TwoCropsTransform


from collections import Counter
#from sklearn.preprocessing._data import StandardScaler
from imblearn.datasets import make_imbalance

#####################################################################
def get_param_cifar10(args):
    # arg
    database_cifar10 = os.path.join(args.dataset_dir,"CIFAR10/")
    cifar10_path = "cifar-10-batches-py/"
    return database_cifar10, cifar10_path

f_d_1 = "data_batch_1"
f_d_2 = "data_batch_2"
f_d_3 = "data_batch_3"
f_d_4 = "data_batch_4"
f_d_5 = "data_batch_5"
f_t = "test_batch"
f_meta = "batches.meta"

filename_list_train = [f_d_1, f_d_2, f_d_3, f_d_4, f_d_5]
filename_test = f_t

#####################################################################
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

#####################################################################
def unpickle_data(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

#####################################################################
# get_meta
def get_meta(database_cifar10, cifar10_path):
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog'csv, 'horse', 'ship', 'truck']
    file_meta = os.path.join(database_cifar10, cifar10_path, "batches_meta.csv")
    if os.path.exists(file_meta):
        print('read batches_meta.csv')
        d_meta = pd.read_csv(file_meta)
    else:
        dict_meta = unpickle(f_meta)
        d_meta = pd.DataFrame.from_dict(dict_meta['label_names'])
        d_meta.to_csv(file_meta, index=False)
    return d_meta

#####################################################################
# get_data_fn_label(f_d_1)
def get_data_fn_label(database_cifar10, cifar10_path, filename):
    # load file
    filename_full = os.path.join(database_cifar10, cifar10_path, filename)
    data_batch = unpickle_data(filename_full)
    
    # get labels
    file_y = os.path.join(database_cifar10, cifar10_path, "_".join(["y", filename]) + ".csv")
    if os.path.exists(file_y):
        print('read y_{}.csv'.format(filename))
        y = pd.read_csv(file_y)
    else:
        y = pd.DataFrame.from_dict(data_batch[b'labels'])
        y.to_csv(file_y, index=False)
    
    counter_y = []
    for i in range(len(y)):
        counter_y.append(y.iloc[i,0])
    print('Counter y for {}:{}'.format(filename, Counter(counter_y)))
    
    
    # get data
    file_data = os.path.join(database_cifar10, cifar10_path, filename + ".csv")
    if os.path.exists(file_data):
        print('read {}.csv'.format(filename))
        d_df = pd.read_csv(file_data)
    else:
        d_df = pd.DataFrame.from_dict(data_batch[b'data'])
        d_df.to_csv(file_data, index=False)

    # get filename
    file_name = os.path.join(database_cifar10, cifar10_path, "_".join(["fn", filename]) + ".csv")
    if os.path.exists(file_name):
        d_fn = pd.read_csv(file_name)
    else:
        fn = data_batch[b'filenames']
        fn_de = []
        for i in range(len(fn)):
            fn_de.append(fn[i].decode("utf-8"))
        d_fn = pd.DataFrame(fn_de)
        d_fn.to_csv(file_name, index=False)
    
    return y.values, d_df.values, d_fn.values


#####################################################################
# train_dataset: 3072 = 1024+1024+1024 -> 32*32 + 32*32 + 32*32
# split one img into two party (32*16*2)
def split_dataset_twoparties(dataset):
    dataset = dataset.values
    dataset = dataset.reshape(len(dataset),3,32,32)
   
    # Transpose the whole data
    dataset = dataset.transpose(0,2,3,1)
    dataset_Xa = dataset[:,0:16,:,:]
    
    dataset_Xb = dataset[:,16:32,:,:]
    
    #####################################################
    dataset_Xa_transpose = dataset_Xa.transpose(0, 3, 1, 2)
    dataset_Xb_transpose = dataset_Xb.transpose(0, 3, 1, 2)

        
    split_Xa_tras_reshape = dataset_Xa_transpose.reshape(len(dataset_Xa_transpose),1536)
    split_Xb_tras_reshape = dataset_Xb_transpose.reshape(len(dataset_Xb_transpose),1536)   
    
    #return dataset_Xa, dataset_Xb
    return split_Xa_tras_reshape, split_Xb_tras_reshape


#####################################################################
# load data and StandardScaler data
# get data from batch files - labels, image data, image file_name -str 
class cifar10():
    def __init__(self, filename_list, args):
        # loading ..... data
        print(' @@@@@@ loading....data.....@@@@@@')
        database_cifar10, cifar10_path = get_param_cifar10(args)
        labels = []
        data = []
        
        # train
        # filename_list for train is a list and len is 5
        if isinstance(filename_list, list):
            for i in range(len(filename_list)):
                label, d_df, d_fn = get_data_fn_label(database_cifar10, cifar10_path, filename_list[i])
                labels.append(pd.DataFrame(label))
                data.append(pd.DataFrame(d_df))
                
            y = pd.concat(labels, axis = 0)
            y = y.reset_index(drop=True)
            img_data = pd.concat(data, axis = 0)
            img_data = img_data.reset_index(drop=True)
            cifar10_y = []
            for i in range(y.shape[0]):
                cifar10_y.append(y.iloc[i,0])
            print('Train_dataset :{}'.format(Counter(cifar10_y)))
            
            # imbalance dataset for experimental requirements
            img_data_res, y_res = make_imbalance(img_data, cifar10_y, sampling_strategy={0:5000, 1:5000, 2:4500, 3:4500, 4:4000, 5:4000, 6:3500, 7:1000, 8:500, 9:250}, random_state=42)
            print('hi, successful imbalancing.....')
            print('Distribution after imbalancing:{}'.format(Counter(y_res)))   
            y = pd.DataFrame(y_res).values

        # test
        else:
            print(' this is test file .....')
            y, img_data, d_fn = get_data_fn_label(database_cifar10, cifar10_path, filename_list)
            img_data = pd.DataFrame(img_data)
            
            # Counter(y)
            cifar10_y = []
            for i in range(len(y)):
                cifar10_y.append(y[i][0])
            print('Test_dataset:{}'.format(Counter(cifar10_y)))            
            
            # imbalance dataset for experimental requirements
            img_data_res, y_res = make_imbalance(img_data, cifar10_y, sampling_strategy={0:1000, 1:1000, 2:800, 3:800, 4:500, 5:500, 6:250, 7:250, 8:100, 9:50}, random_state=42)
            print('hi, successful imbalancing.....')
            print('Distribution after imbalancing:{}'.format(Counter(y_res)))     

            y = pd.DataFrame(y_res).values    
           
        self.img_data = img_data_res
        self.Xa, self.Xb = split_dataset_twoparties(self.img_data)        
        print('type of self.Xa:{} and shape:{}'.format(type(self.Xa), self.Xa.shape))
        self.x = [self.Xa, self.Xb]
        self.y = y
        print('{}: self.y - type:, shape{}'.format(type(self.y), self.y.shape))
        
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, indexx):
        data = []
        labels = []
        for i in range(2):
            data.append(self.x[i][indexx])
        labels.append(self.y[indexx])
        return data, np.array(labels).ravel()
    
    
#####################################################################
# input:dataset type:<class 'data.cifar10.cifar10'>
# output:
class cifar10Aug():
    def __init__(self, dataset, args):
        # loading ..... data
        print(' @@@@@@ loading....data.....@@@@@@')
        print('dataset type:{}'.format(type(dataset)))
        
        self.Xa = copy.deepcopy(dataset.Xa)
        self.Xb = copy.deepcopy(dataset.Xb)
        self.y = copy.deepcopy(dataset.y)
        print('dataset.Xa shape:{}, type:{}'.format(self.Xa.shape, type(self.Xa)))
        print('dataset.Xb shape:{}, type:{}'.format(self.Xb.shape, type(self.Xb)))
        
        self.transform = TwoCropsTransform()
        
        self.x = [self.Xa, self.Xb]        
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, indexx):
        data = []
        labels = []
        for i in range(2):
            x = self.x[i][indexx]
            if self.transform is not None:
                x = x.reshape(3, 16, 32)
                x = x.transpose(1,2,0)
                x = self.transform(x)
            data.append(x)
        labels.append(self.y[indexx])
        return data, np.array(labels).ravel()



#####################################################################
def load_dataset_cifar10(args, model_type):
    train_dataset = None
    test_dataset = None
    train_dataset_aug = None
    test_dataset_aug = None

    """ step 1: get_dataset/get_dataset_aug, len and indices """
    train_dataset = cifar10(filename_list_train, args)
    test_dataset = cifar10(filename_test, args)

    # indices of train and test datasets
    n_train = len(train_dataset)
    n_test = len(test_dataset)
    train_indices = list(range(n_train))
    test_indices = list(range(n_test))

    # shuffle train samples
    random.shuffle(train_indices)
    
    logging.info("***** train/test data num： {}, {}".format(len(train_indices), len(test_indices)))
    
    # get_dataset_aug
    if model_type == 'pretrain':
        print('............ this is pretrain........')
        # train_datasest_augumentation
        #train_dataset_aug = cifar10Aug(filename_list_train, args)
        train_dataset_aug = cifar10Aug(train_dataset, args)
        print('train_dataset_aug len:{}'.format(len(train_dataset_aug)))

        # test_datasest_augumentation
        #test_dataset_aug = cifar10Aug(filename_test, args)
        test_dataset_aug = cifar10Aug(test_dataset, args)
        print('test_dataset_aug len:{}'.format(len(test_dataset_aug)))

    
    """ step 2: get aligned labeled sampler (indices) , test sampler (indices) , train_local_sampler (indices) """
    train_aligned_labeled_num = int(n_train * args.aligned_samples_percent * args.labeled_samples_percent)
    train_aligned_labeled_indices = train_indices[:train_aligned_labeled_num]
    
    # all samples are local - n_train; train_indices
    logging.info("***** train_aligned_labeled_num:{}; train_local_num (all local train datasets):{}".format(train_aligned_labeled_num, n_train))

    train_aligned_labeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_aligned_labeled_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
    # all train samples are local train samples
    train_local_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    

    """ step 3: load dataset with torch DataLoader"""
    train_aligned_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_aligned_labeled_sampler, pin_memory=False, drop_last=False)
    train_local_loader = torch.utils.data.DataLoader(train_dataset_aug, batch_size=args.batch_size, sampler=train_local_sampler, pin_memory=False, drop_last=False)
    
    # torch DataLoader; pretrain  ->test_dataset + test_dataset_aug;
    if model_type == 'pretrain':
        print('...............this is pretrain................')
        test_loader = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, pin_memory=False, drop_last=False),
                       torch.utils.data.DataLoader(test_dataset_aug, batch_size=args.batch_size, sampler=test_sampler, pin_memory=False, drop_last=False)]
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, pin_memory=False, drop_last=False)
    
    assert train_aligned_loader is not None or test_loader is not None or train_local_loader is not None, print('invalid dataloader!')
    
    """ step 4: return """
    return train_aligned_loader, train_local_loader, test_loader, args
