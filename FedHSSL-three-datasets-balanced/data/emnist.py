import os
import pandas as pd
import numpy as np
import copy
import random
import logging

from scipy.io import loadmat
from collections import Counter
from sklearn.preprocessing._data import StandardScaler
import sys
sys.path.append(os.getcwd())
from data.augmentation_img import TwoCropsTransform_emnist
from PIL import Image

import torch

#####################################################################
def get_param_emnist(args):
    # arg
    database_EMNIST = os.path.join(args.dataset_dir, "EMNIST/")
    return database_EMNIST

# bymerge -> 814,255 characters. 47 unbalanced classes
# 28*28 image 
matlab_path = "matlab/"
file_bymerge = "emnist-bymerge.mat"

#####################################################################
# bymerge -> type: dict
def load_file(database_EMNIST):
    file_name = os.path.join(database_EMNIST, matlab_path,file_bymerge)
    bymerge = loadmat(file_name)
    return bymerge

#####################################################################
# get emnist-bymerge.mat - "images", "labels" - imbalanced
# dtype: "Train", "Test"
# ftype: "images", "labels"
# examples:
# img_train_df = get_features("train","images")
# img_test_df = get_features("test","images")
# label_train_df = get_features("train","labels")
# label_test_df = get_features("test","labels")
def get_features(database_EMNIST, dtype, ftype):
    f_bymerge = os.path.join(database_EMNIST, matlab_path, "_".join(["f_bymerge", dtype, ftype]) + ".csv")
    if os.path.exists(f_bymerge):
        print("read " + f_bymerge)
        df_f = pd.read_csv(f_bymerge)
    else:
        bymerge = load_file(database_EMNIST)
        df_f = pd.DataFrame(bymerge['dataset'][dtype][0,0][ftype][0,0])
        df_f.to_csv(f_bymerge, index=False)
    return df_f

#####################################################################
def get_labels_train(database_EMNIST, dtype, args):
    # labels -> data
    labels_df = get_features(database_EMNIST, dtype, "labels")
    
    #train Counter({1: 38304, 7: 36020, 3: 35285, 0: 34618, 2: 34307, 6: 34150, 8: 33924, 9: 33882, 4: 33656, 5: 31280, 24: 27664, 39: 24657, 28: 23509, 21: 20381, 46: 18248, 30: 15388, 18: 14733, 45: 14060, 12: 12963, 22: 11612, 43: 11444, 25: 10748, 38: 10152, 36: 10009, 29: 9766, 15: 9098, 42: 8682, 23: 8237, 31: 7588, 32: 7403, 34: 7092, 10: 6411, 19: 5689, 33: 5598, 35: 5416, 37: 5080, 27: 5047, 20: 4998, 14: 4925, 13: 4606, 11: 3874, 41: 3693, 17: 3097, 44: 2966, 26: 2603, 40: 2535, 16: 2534})
    #test {1: 6400, 7: 5873, 3: 5827, 6: 5787, 2: 5765, 0: 5745, 8: 5655, 9: 5651, 4: 5498, 5: 5326, 24: 4690, 39: 4066, 28: 3899, 21: 3358, 46: 2979, 30: 2528, 18: 2413, 45: 2365, 12: 2156, 22: 1984, 43: 1872, 25: 1812, 38: 1708, 36: 1668, 29: 1630, 42: 1535, 15: 1524, 23: 1351, 32: 1262, 31: 1223, 34: 1195, 10: 1058, 37: 932, 35: 925, 19: 912, 33: 897, 14: 860, 27: 835, 20: 809, 13: 735, 11: 652, 41: 583, 17: 576, 44: 533, 16: 430, 40: 426, 26: 415}
    labels_counter = Counter(labels_df.iloc[:,0])
    # Order the dictionary by values in descending order
    sorted_dict = dict(sorted(labels_counter.items(), key=lambda item: item[1], reverse=True))
    
    # get the first 5 and last 5 classes -> 10 classes -> imbalanced
    first_5_keys_sorted = list(sorted_dict.keys())[:5]
    last_5_keys_sorted = list(sorted_dict.keys())[-5:]
    # labels_to_match_sorted:[1, 7, 3, 0, 2, 17, 44, 26, 40, 16]
    labels_to_match_sorted = first_5_keys_sorted + last_5_keys_sorted
    print('labels_to_match:{}'.format(labels_to_match_sorted))    
    args.mul_classes = labels_to_match_sorted

    # select labels according to labels_to_match_sorted
    labels_selected_train = labels_df[labels_df['0'].isin(labels_to_match_sorted)]  
    
    # get indices of labels_selected_train
    labels_indices_train = labels_selected_train.index.tolist()
    
    # Counter after labels selected
    # After labels selected_train:
    # Counter({1: 38304, 7: 36020, 3: 35285, 0: 34618, 2: 34307, 17: 3097, 44: 2966, 26: 2603, 40: 2535, 16: 2534})
    print('After labels selected_{}:{}'.format(dtype, Counter(labels_selected_train.iloc[:,0])))
        
    return labels_indices_train, labels_selected_train, args


#####################################################################
def get_labels_test(database_EMNIST, dtype, args):
    # labels -> data
    labels_df = get_features(database_EMNIST, dtype, "labels")
    
    print('args.mul_classes":"{}'.format(args.mul_classes))
    #test:Counter({1: 6400, 7: 5873, 3: 5827, 6: 5787, 2: 5765, 0: 5745, 8: 5655, 9: 5651, 4: 5498, 5: 5326, 24: 4690, 39: 4066, 28: 3899, 21: 3358, 46: 2979, 30: 2528, 18: 2413, 45: 2365, 12: 2156, 22: 1984, 43: 1872, 25: 1812, 38: 1708, 36: 1668, 29: 1630, 42: 1535, 15: 1524, 23: 1351, 32: 1262, 31: 1223, 34: 1195, 10: 1058, 37: 932, 35: 925, 19: 912, 33: 897, 14: 860, 27: 835, 20: 809, 13: 735, 11: 652, 41: 583, 17: 576, 44: 533, 16: 430, 40: 426, 26: 415})

    # select labels(y) according to labels_to_match_sorted
    # labels_to_match_sorted:[1, 7, 3, 0, 2, 17, 44, 26, 40, 16]
    labels_selected_test = labels_df[labels_df['0'].isin(args.mul_classes)]  
    
    # get indices of labels_selected_test
    labels_indices_test = labels_selected_test.index.tolist()
    
    # Counter after labels selected
    # After labels selected_test:
    # Counter({1: 6400, 7: 5873, 3: 5827, 2: 5765, 0: 5745, 17: 576, 44: 533, 16: 430, 40: 426, 26: 415})
    print('After labels selected_{}:{}'.format(dtype, Counter(labels_selected_test.iloc[:,0])))
        
    return labels_indices_test, labels_selected_test


#####################################################################
def split_dataset_twoparties(dataset):
    # 28*28 = 784 / split data into two parties -> up and down
    dataset = pd.DataFrame(dataset)
    dataset_Xa = dataset.iloc[:,0:392]
    dataset_Xb = dataset.iloc[:,392:784]
    return dataset_Xa.values, dataset_Xb.values


#####################################################################
# dataset -> emnist10classes2parties
# load data and StandardScaler data
class EMNIST():
    def __init__(self, dtype, args):
        # loading ......data
        print(' @@@@@@ loading....data.....@@@@@@ ')
        self.args = args
        dir_dataset_NUSWIDE = get_param_emnist(self.args)
        mul_classes = self.args.mul_classes
        self.dtype = dtype
        # labels
        if dtype == 'train':
            # train labels -> labels_selected_train
            # labels_to_match_sorted:[1, 7, 3, 0, 2, 17, 44, 26, 40, 16]
            # After labels selected_train:Counter({1: 38304, 7: 36020, 3: 35285, 0: 34618, 2: 34307, 17: 3097, 44: 2966, 26: 2603, 40: 2535, 16: 2534})
            labels_indices, labels_selected, self.args = get_labels_train(dir_dataset_NUSWIDE, dtype, self.args)
            
            # data -> 784 = 28*28       
            img_df = get_features(dir_dataset_NUSWIDE, dtype, "images")
            img_selected =  img_df.iloc[labels_indices]
        elif dtype == 'test':
            print('args.mul_classes:{}'.format(self.args.mul_classes))
            labels_indices, labels_selected = get_labels_test(dir_dataset_NUSWIDE, dtype, self.args)
            img_df = get_features(dir_dataset_NUSWIDE, dtype, "images")
            img_selected = img_df.iloc[labels_indices]
        
        print('data shape: {} and type:{}'.format(img_selected.shape, type(img_selected)))            
        
        self.img = img_selected
        self.Xa, self.Xb = split_dataset_twoparties(self.img)

        self.x = [self.Xa, self.Xb]
        
        # mapping classes: [1, 7, 3, 0, 2, 17, 44, 26, 40, 16] to [1, 4, 3, 0, 2, 5, 6, 7, 8, 9]
        mapping = {7:4, 17:5, 44:6, 26:7, 40:8, 16:9}
        labels_selected.loc[:,'0'] = labels_selected['0'].replace(mapping)
        print('After labels mapping_{}:{}'.format(dtype, Counter(labels_selected.iloc[:,0])))
        self.y = labels_selected.values


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
class EMNISTAug():
    def __init__(self, dataset, args):
        # loading ..... data
        print(' @@@@@@ loading....data.....@@@@@@')
        self.args = args
        
        self.Xa = copy.deepcopy(dataset.Xa)
        self.Xb = copy.deepcopy(dataset.Xb)
        self.y = copy.deepcopy(dataset.y)
        
        self.transform = TwoCropsTransform_emnist()

        self.x = [self.Xa, self.Xb]        
        
    def __len__(self):
        return len(self.y)     
    
    def __getitem__(self, indexx):
        data = []
        labels = []
        for i in range(2):
            x = self.x[i][indexx]
            if self.transform is not None:
                x = x.reshape(14, 28)
                x = Image.fromarray(x,'L')
                x = self.transform(x)
            data.append(x)
        labels.append(self.y[indexx])
        return data, np.array(labels).ravel()
    
#####################################################################    
def load_dataset_emnist(args, model_type):
    train_dataset = None
    test_dataset = None
    train_dataset_aug = None
    test_dataset_aug = None

    """ step 1: get_dataset/get_dataset_aug, len and indices """
    train_dataset = EMNIST("train", args)
    print('len of train_dataset:{}'.format(len(train_dataset)))
    test_dataset = EMNIST("test", train_dataset.args)    
    print('len of test_dataset:{}'.format(len(test_dataset)))               

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
        train_dataset_aug = EMNISTAug(train_dataset, args)
        print('train_dataset_aug len:{}'.format(len(train_dataset_aug)))

        # test_datasest_augumentation
        test_dataset_aug = EMNISTAug(test_dataset, train_dataset_aug.args)
        print('test_dataset_aug len:{}'.format(len(test_dataset_aug)))
    
    """ step 2: get aligned labeled sampler (indices) , test sampler (indices) , train_local_sampler (indices) """
    # aligned samples - default is 20% of all samples
    # labeled samples - default is 100% of aligned samples
    train_aligned_labeled_num = int(n_train * args.aligned_samples_percent * args.labeled_samples_percent)
    train_aligned_labeled_indices = train_indices[:train_aligned_labeled_num]
    
    # all samples are local - n_train; train_indices
    logging.info("***** train_aligned_labeled_num:{}; train_local_num (all local train datasets):{}".format(train_aligned_labeled_num, n_train))

    # sampler -> defines the strategy to draw samples from the dataset.
    # train_aligned_labeled_indices is a shuffle sequence of indices;
    # train_aligned_labeled_sampler: Samples' indices randomly from train_aligned_labeled_indices, without replacement.
    train_aligned_labeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_aligned_labeled_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
    # all train samples are local train samples
    train_local_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    
    """ step 3: load dataset with torch DataLoader"""
    # torch DataLoader; test_load -> test_dataset, test_dataset_aug
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
