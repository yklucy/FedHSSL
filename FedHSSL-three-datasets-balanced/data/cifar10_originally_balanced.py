import pickle
import os
import sys
import pandas as pd
import random
import logging
import copy
import torch
sys.path.append(os.getcwd())
from collections import Counter
import numpy as np


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

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def unpickle_data(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


# get_meta
def get_meta(database_cifar10, cifar10_path):
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    file_meta = os.path.join(database_cifar10, cifar10_path, "batches_meta.csv")
    if os.path.exists(file_meta):
        print('read batches_meta.csv')
        d_meta = pd.read_csv(file_meta)
    else:
        dict_meta = unpickle(f_meta)
        d_meta = pd.DataFrame.from_dict(dict_meta['label_names'])
        d_meta.to_csv(file_meta, index=False)
        
    #print('d_meta:{} \t'.format(d_meta))
    return d_meta


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

def split_dataset_twoparties(dataset):
    dataset = pd.DataFrame(dataset)

    dataset_p1_1 = dataset.iloc[:,0:512]
    dataset_p1_2 = dataset.iloc[:,1024:1536]
    dataset_p1_3 = dataset.iloc[:,2048:2560]
    
    dataset_p2_1 = dataset.iloc[:,512:1024]
    dataset_p2_2 = dataset.iloc[:,1536:2048]
    dataset_p2_3 = dataset.iloc[:,2560:3072]
    
    dataset_Xa = pd.concat([dataset_p1_1, dataset_p1_2, dataset_p1_3], axis=1)
    dataset_Xb = pd.concat([dataset_p2_1, dataset_p2_2, dataset_p2_3], axis=1)

    return dataset_Xa.values, dataset_Xb.values


# load data and StandardScaler data
class cifar10():
    def __init__(self, filename_list, args):
        # loading ..... data
        print(' @@@@@@ loading....data.....@@@@@@')
        database_cifar10, cifar10_path = get_param_cifar10(args)
        labels = []
        data = []
        
        # train
        if isinstance(filename_list, list):
            for i in range(len(filename_list)):
                label, d_df, d_fn = get_data_fn_label(database_cifar10, cifar10_path, filename_list[i])
                labels.append(pd.DataFrame(label))
                data.append(pd.DataFrame(d_df))
                
            y = pd.concat(labels, axis = 0)
            y = y.reset_index(drop=True)
            img_data = pd.concat(data, axis = 0)
            img_data = img_data.reset_index(drop=True)
            
            #length = y.shape[0]; Counter(y)
            cifar10_train_y = []
            for i in range(y.shape[0]):
                cifar10_train_y.append(y.iloc[i,0])
            # type of cifar10_y:<class 'list'>
            print('type of cifar10_y:{}'.format(type(cifar10_train_y)))
            # Train_dataset :Counter({6: 5000, 9: 5000, 4: 5000, 1: 5000, 2: 5000, 7: 5000, 8: 5000, 3: 5000, 5: 5000, 0: 5000})
            print('Train_dataset :{}'.format(Counter(cifar10_train_y)))
            
            y = y.values
        # test
        else:
            y, img_data, d_fn = get_data_fn_label(database_cifar10, cifar10_path, filename_list)
            # Counter(y)
            cifar10_test_y = []
            for i in range(len(y)):
                cifar10_test_y.append(y[i][0])
            print('Test_dataset:{}'.format(Counter(cifar10_test_y)))

        self.img_data = img_data
        # split dataset into two parties

        self.Xa, self.Xb = split_dataset_twoparties(self.img_data)
        print('shape self.Xa:{}'.format(self.Xa.shape))
        print('shape self.Xb:{}'.format(self.Xb.shape))
        self.x = [self.Xa, self.Xb]
        
        self.y = y
        print('{}: self.y - type:, shape{}'.format(type(self.y), self.y.shape))
        print('self.y len:{}'.format(y.shape[0]))


    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, indexx):
        data = []
        labels = []
        for i in range(2):
            data.append(self.x[i][indexx])
        labels.append(self.y[indexx])
        return data, np.array(labels).ravel()
    

# FedHSSL Augumentation Transform
class cifar10_TabularDataTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, percent=0.3, params=None):
        self.percent = percent
        self.params = params
        print('hi, I am TabularDataTransform.__init__!')

    def __call__(self, x, client_id=0):
        masked_dim = int(x.shape[-1]*self.percent)
        masked_index = np.random.choice(x.shape[-1], masked_dim, replace=False)
        q = copy.deepcopy(x)
        masked_value = np.random.uniform(self.params[client_id][0][masked_index], self.params[client_id][1][masked_index])
        q[masked_index] = masked_value
        return [x, q]


class cifar10Aug():
    def __init__(self, filename_list, args):
        # loading ..... data
        print(' @@@@@@ loading....data.....@@@@@@')
        dataset = cifar10(filename_list, args)
        self.Xa = dataset.Xa
        self.Xb = dataset.Xb
        self.y = dataset.y
        
        self.Xa_min = np.min(self.Xa, axis=0)
        self.Xa_max = np.max(self.Xa, axis=0)
        self.Xb_min = np.min(self.Xb, axis=0)
        self.Xb_max = np.max(self.Xb, axis=0)
        self.params = [[self.Xa_min, self.Xa_max], [self.Xb_min, self.Xb_max]]
        
        self.transform = cifar10_TabularDataTransform(0.3, self.params)

        self.x = [self.Xa, self.Xb]        
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, indexx):
        data = []
        labels = []
        for i in range(2):
            x = self.x[i][indexx]
            if self.transform is not None:
                x = self.transform(x, i)
            data.append(x)
        labels.append(self.y[indexx])
        return data, np.array(labels).ravel()



def load_dataset_cifar10(args, model_type):
    train_dataset = None
    test_dataset = None
    train_dataset_aug = None
    test_dataset_aug = None

    """ step 1: get_dataset/get_dataset_aug, len and indices """
    train_dataset = cifar10(filename_list_train, args)
    test_dataset = cifar10(filename_test, args)
    print('shape train_dataset Xa:{} and type:{}'.format(train_dataset.Xa.shape, type(train_dataset.Xa)))
    print('shape train_dataset Xb:{} and type:{}'.format(train_dataset.Xb.shape, type(train_dataset.Xb)))
    print('shape test_dataset Xa:{} and type:{}'.format(test_dataset.Xa.shape, type(test_dataset.Xa)))
    print('shape test_dataset Xb:{} and type:{}'.format(test_dataset.Xb.shape, type(test_dataset.Xb)))

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
        train_dataset_aug = cifar10Aug(filename_list_train, args)
        print('train_dataset_aug len:{}'.format(len(train_dataset_aug)))

        # test_datasest_augumentation
        test_dataset_aug = cifar10Aug(filename_test, args)
        print('test_dataset_aug len:{}'.format(len(test_dataset_aug)))
        print('shape train_dataset_aug Xa:{} and type:{}'.format(train_dataset_aug.Xa.shape, type(train_dataset_aug.Xa)))
        print('shape train_dataset_aug Xb:{} and type:{}'.format(train_dataset_aug.Xb.shape, type(train_dataset_aug.Xb)))
        print('shape test_dataset_aug Xa:{} and type:{}'.format(test_dataset_aug.Xa.shape, type(test_dataset_aug.Xa)))
        print('shape test_dataset_aug Xb:{} and type:{}'.format(test_dataset_aug.Xb.shape, type(test_dataset_aug.Xb)))
    
    """ step 2: get aligned labeled sampler (indices) , test sampler (indices) , train_local_sampler (indices) """
    # aligned samples - default is 20% of all samples
    # labeled samples - default is 100% of aligned samples
    train_aligned_labeled_num = int(n_train * args.aligned_samples_percent * args.labeled_samples_percent)
    train_aligned_labeled_indices = train_indices[:train_aligned_labeled_num]
    
    # all samples are local - n_train; train_indices
    logging.info("***** train_aligned_labeled_num:{}; train_local_num (all local train datasets):{}".format(train_aligned_labeled_num, n_train))

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
