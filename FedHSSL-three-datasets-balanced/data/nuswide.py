import os
import logging
import sys

import numpy as np
import pandas as pd
import copy
import random
import torch.utils
sys.path.append(os.getcwd())

from sklearn.preprocessing._data import StandardScaler

#####################################################################
def get_param_nuswide(args):
    # arg
    dir_dataset_NUSWIDE = os.path.join(args.dataset_dir, "NUS-WIDE/")
    return dir_dataset_NUSWIDE

feature_path = "Low_Level_Features/"
tag_path = "NUS_WID_Tags/"
label_path = "Groundtruth/TrainTestLabels/"
concepts_path = "ConceptsList/"


#####################################################################
# 10 classes
#mul_classes = ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']

# get concepts: 81*1 -> 81 rows and 1 column
# example:
# get_concepts()
# selected_labels = get_concepts().values.ravel()
def get_concepts(dir_dataset_NUSWIDE):
    f_concepts = os.path.join(dir_dataset_NUSWIDE, concepts_path, "Concepts81.csv")
    if os.path.exists(f_concepts):
        print("read Concepts.csv")
        concepts_data = pd.read_csv(f_concepts)
    else:
        c_file = os.path.join(dir_dataset_NUSWIDE, concepts_path, "Concepts81.txt")
        concepts_data = pd.read_csv(c_file, header=None, sep=" ")
        concepts_data.dropna(axis=1, inplace=True)
        concepts_data.to_csv(f_concepts, index=False)
    return concepts_data


#####################################################################
# get labels from Groundtruth .txt files, 10 concepts/columns
# example:
# selected_labels =  ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake'] 
# labels_train = get_labels(selected_labels, "Train")
# labels_test = get_labels(selected_labels, "Test")
def get_labels(selected_labels, dtype, dir_dataset_NUSWIDE):
    f_labels = os.path.join(dir_dataset_NUSWIDE, label_path, "_".join(["y_labels", dtype]) + ".csv")
    if os.path.exists(f_labels):
        print("read " + os.path.join("_".join(["y_labels", dtype]) + ".csv"))
        labels_data = pd.read_csv(f_labels)
    else:
        d_labels = []
        for label in selected_labels:
            l_file = os.path.join(dir_dataset_NUSWIDE, label_path, "_".join(["Labels", label, dtype]) + ".txt")
            df = pd.read_csv(l_file, header=None)
            df.columns = [label]
            d_labels.append(df)
        labels_data = pd.concat(d_labels, axis=1)
        labels_data.to_csv(f_labels, index=False)
    return labels_data

#####################################################################
# FedHSSL: select labels
# FedHSSL: to get the sum of columns use axis=1, selecting smaples with only one label for each row
# FedHSSL: for example, if a row has two or more labels, the sum result will be greater than 1, so do not select it.
def select_label(mul_classes, dtype, labels_data, dir_dataset_NUSWIDE):
    if len(mul_classes) > 1:
        selected = labels_data[labels_data.sum(axis=1) == 1]
    else:
        selected = labels_data
    print('selected_label.shape: {}'.format(selected.shape))
    selected.to_csv(os.path.join(dir_dataset_NUSWIDE, label_path, "_".join([dtype,"selected_lables_FedHSSL"])+".csv"), index=False)

    return selected


#####################################################################
# get image low_level_features: CH, CM55, CORR, EDH, WT
# named XA -> 634 columns
# dtype: "Train", "Test"
# examples:
# features_train_selected = get_features("Train", selected)
# features_test_selected = get_features("Test", selected)
def get_features(dtype, selected, dir_dataset_NUSWIDE):
    f_features = os.path.join(dir_dataset_NUSWIDE, feature_path, "_".join(["X_features", dtype])+".csv")
    if os.path.exists(f_features):
        print("read " + os.path.join("_".join(["X_features", dtype])+".csv"))
        features_data_selected = pd.read_csv(f_features)
    else:
        df = []
        for filename in os.listdir(os.path.join(dir_dataset_NUSWIDE, feature_path)):
            if (filename.startswith(dtype)):
                f_file = os.path.join(dir_dataset_NUSWIDE, feature_path, filename)
                f_df = pd.read_csv(f_file, header=None, sep=" ")
                f_df.dropna(axis=1, inplace=True)
                df.append(f_df)
        features_data = pd.concat(df, axis=1)
        
        # selected samples
        features_data_selected = features_data.loc[selected.index]
        features_data_selected.to_csv(f_features, index=False)
    return features_data_selected


#####################################################################
# get tags
# named XB -> 1000 columns
# dtype: "Train", "Test"
# ttype: "Tags1k", "Tags81"
# ftype: ".dat", ".txt"
# t_sep: "\t", " "
# examples:
# tags_train = get_tags("Train", "Tags1k", ".dat", " ", selected_tags_test)
# tags_test = get_tags("Test", "Tags1k", ".dat", " ", selected_tags_test)
def get_tags(dtype, ttype, ftype, t_sep, selected, dir_dataset_NUSWIDE):
    f_tags = os.path.join(dir_dataset_NUSWIDE, tag_path, "_".join(["X", ttype, dtype])+".csv")
    if os.path.exists(f_tags):
        print("read " + os.path.join("_".join(["X_tags", dtype])+".csv"))
        tags_data_selected = pd.read_csv(f_tags)
    else:
        t_file = os.path.join(dir_dataset_NUSWIDE, tag_path, "_".join([dtype, ttype])+ftype)
        tags_data = pd.read_csv(t_file, header=None, sep=t_sep)
        tags_data.dropna(axis=1, inplace=True)
        
        tags_data_selected = tags_data.loc[selected.index]
        
        tags_data_selected.to_csv(f_tags, index=False)
    return tags_data_selected


#####################################################################
# transfer y from OneHot to Category -> one column (y) with 10 classes
def label_trans(y):
    y_ = []
    pos_count = 0
    neg_count = 0
    count = {}
    # y is an array
    for i in range(y.shape[0]):
        # get the index of the nonzero label
        # when getting labels, just select samples with only one label, so there is the only nonzero index for each row. 
        # transform OneHot to category - > one column (y) with 10 classes
        label = np.nonzero(y[i,:])[0][0]
        y_.append(label)
        if label not in count:
            count[label] = 1
        else:
            count[label] = count[label] + 1
    logging.info("***** Counter:{}".format(count))
    
    y = np.expand_dims(y_, axis=1)
    
    return y

#####################################################################
# dataset -> nuswide10classes2parties
# load data and StandardScaler data
class NUSWIDE():
    def __init__(self, dtype, args):
        # loading ..... data
        print(' @@@@@@ loading....data.....@@@@@@')
        dir_dataset_NUSWIDE = get_param_nuswide(args)
        mul_classes = args.mul_classes
        # labels
        self.labels = get_labels(mul_classes, dtype, dir_dataset_NUSWIDE)
        
        # selected labels
        self.selected_label = select_label(mul_classes, dtype, self.labels, dir_dataset_NUSWIDE)

        # selected features - Xa [634]
        self.selected_features = get_features(dtype, self.selected_label, dir_dataset_NUSWIDE)
    
        # selected tags - Xb [1000]
        self.selected_tags = get_tags(dtype, "Tags1k", ".dat", "\t", self.selected_label, dir_dataset_NUSWIDE)
        
        print(' @@@@@@ StandardScaler....data.....@@@@@@')
        # StandardScaler data
        self.data_scaler_model = StandardScaler()
        self.Xa = self.data_scaler_model.fit_transform(self.selected_features.values)
        self.Xb = self.data_scaler_model.fit_transform(self.selected_tags.values)
        self.x = [self.Xa, self.Xb]
        self.y = label_trans(self.selected_label.values)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, indexx):
        data = []
        labels = []
        
        # x[0] -> Xa; x[1] -> Xb
        for i in range(2):
            # read the indexx line 
            data.append(self.x[i][indexx])
        labels.append(self.y[indexx])
        return data, np.array(labels).ravel()


#####################################################################
# FedHSSL Augumentation Transform
class TabularDataTransform:
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

#####################################################################
# dataset augmentation -> nuswide10classes2parties - Xa = [Xa, Xa~]; Xb = [Xb, Xb~]
class NUSWIDEAug():
    def __init__(self, dataset, args):
        dir_dataset_NUSWIDE = get_param_nuswide(args)
        mul_classes = args.mul_classes
        self.Xa = copy.deepcopy(dataset.Xa)
        self.Xb = copy.deepcopy(dataset.Xb)
        self.y = copy.deepcopy(dataset.y)
        
        self.Xa_min = np.min(self.Xa, axis=0)
        self.Xa_max = np.max(self.Xa, axis=0)
        self.Xb_min = np.min(self.Xb, axis=0)
        self.Xb_max = np.max(self.Xb, axis=0)
        self.params = [[self.Xa_min, self.Xa_max], [self.Xb_min, self.Xb_max]]

        self.transform = TabularDataTransform(0.3, self.params)

        self.x = [self.Xa, self.Xb]
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        data = []
        labels = []
        for i in range(2):
            x = self.x[i][indexx]
            if self.transform is not None:
                x = self.transform(x, i)
            data.append(x)
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()


#####################################################################
# load dataset according to aligned percent and labeled percent
def load_dataset_nuswide(args, model_type):
    train_dataset = None
    test_dataset = None
    train_dataset_aug = None
    test_dataset_aug = None
    mul_classes = args.mul_classes
    NUM_CLASSES = len(mul_classes)
    
    """ step 1: get_dataset/get_dataset_aug, len and indices """
    train_dataset = NUSWIDE('Train', args)
    test_dataset = NUSWIDE('Test', args)

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
        print('..............type: pretrain!')
        # train_datasest_augumentation
        train_dataset_aug = NUSWIDEAug(train_dataset, args)

        # test_datasest_augumentation
        test_dataset_aug = NUSWIDEAug(test_dataset, args)
    
    
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
    # train_aligned_loader -> train_dataset; 
    # train_local_loader -> train_dataset_aug
    train_aligned_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_aligned_labeled_sampler, pin_memory=False, drop_last=False)
    train_local_loader = torch.utils.data.DataLoader(train_dataset_aug, batch_size=args.batch_size, sampler=train_local_sampler, pin_memory=False, drop_last=False)
    
    # torch DataLoader; test_load -> test_dataset；
    # pretrain -> test_dataset_aug
    if model_type == 'pretrain':
        print('...........type: pretrain!')
        test_loader = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, pin_memory=False, drop_last=False),
                       torch.utils.data.DataLoader(test_dataset_aug, batch_size=args.batch_size, sampler=test_sampler, pin_memory=False, drop_last=False)]
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, pin_memory=False, drop_last=False)
    
    assert train_aligned_loader is not None or test_loader is not None or train_local_loader is not None, print('invalid dataloader!')
    
    """ step 4: return """
    return train_aligned_loader, train_local_loader, test_loader, args


