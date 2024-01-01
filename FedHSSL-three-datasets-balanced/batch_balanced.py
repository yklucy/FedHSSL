import numpy as np
from collections import Counter
import torch
import copy
from PIL import ImageFilter, Image
from torchvision import transforms
import random


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def balance_with_labels(trn_X_in, trn_y_in, num, args):
    print('args.dataset:{}'.format(args.dataset))
    
    # get the label distribution
    target = trn_y_in.view(-1)
    t_dict = Counter(target.numpy().tolist())
    print('t_dict:{}'.format(t_dict))

    Xa = None
    Xb = None
    trn_y = None
    for j in t_dict.keys():
        if t_dict[j] > num:
            data_balancing_Xa = None
            data_balancing_Xb = None
            trn_yy = None
            # undersampling
            # get the index from target
            indices_value_under = (torch.nonzero(trn_y_in == j))[:,0]
            random_indices_under = torch.randperm(len(indices_value_under))[:num]                        
            indices_under = indices_value_under[random_indices_under]

            data_balancing_Xa = trn_X_in[0][indices_under]
            data_balancing_Xb = trn_X_in[1][indices_under]

            trn_yy = torch.full((num,),j)
        else:
            data_balancing_Xa = None
            data_balancing_Xb = None
            trn_yy = None
            # oversampling
            # get the index from target
            indices_value_over = (torch.nonzero(trn_y_in == j))[:,0]
            
            # original data
            data_balancing_Xa = trn_X_in[0][indices_value_over]
            data_balancing_Xb = trn_X_in[1][indices_value_over]
            trn_yy = torch.full((len(indices_value_over),),j)
            for i in range(num-t_dict[j]):    
                random_indices_over = torch.randperm(len(indices_value_over))[:1]
                index_over = indices_value_over[random_indices_over]
                tmp_Xa = trn_X_in[0][index_over]
                tmp_Xb = trn_X_in[1][index_over]
                
                if args.dataset == 'EMNIST':
                    tmp_Xa = tmp_Xa.numpy().reshape(14, 28)
                    tmp_Xb = tmp_Xb.numpy().reshape(14, 28)
                    
                    tmp_Xa = Image.fromarray(tmp_Xa,'L')
                    tmp_Xb = Image.fromarray(tmp_Xb,'L')
                    transform_img_EMNIST = transforms.Compose([
                                                        transforms.RandomResizedCrop((14,28), scale=(0.3, 1.)),
                                                        transforms.RandomApply([
                                                            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
                                                        ], p=0.8),
                                                        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor()
                                                    ])
                    new_Xa = transform_img_EMNIST(tmp_Xa).reshape(1,-1)
                    new_Xb = transform_img_EMNIST(tmp_Xb).reshape(1,-1)
                
                if args.dataset == 'CIFAR10':
                    # image augmentation
                    tmp_Xa = tmp_Xa.numpy().reshape(3, 16, 32)
                    tmp_Xb = tmp_Xb.numpy().reshape(3, 16, 32)

                    tmp_Xa = tmp_Xa.transpose(1,2,0)
                    tmp_Xb = tmp_Xb.transpose(1,2,0)
                    
                    transform_img_CIFAR10 = transforms.Compose([
                                                            transforms.ToPILImage(mode='RGB'),
                                                            transforms.RandomResizedCrop((16,32), scale=(0.3, 1.)),
                                                            transforms.RandomApply([
                                                                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
                                                            ], p=0.8),
                                                            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor()
                                                    ])
                    new_Xa = transform_img_CIFAR10(tmp_Xa).reshape(1,-1)
                    new_Xb = transform_img_CIFAR10(tmp_Xb).reshape(1,-1)
                                        
                if args.dataset == 'NUSWIDE':
                    tmp_Xa = np.array(tmp_Xa) + np.random.normal(0, 0.1, size=tmp_Xa.shape)  # Adding random noise as an example
                    tmp_Xb = np.array(tmp_Xb) + np.random.normal(0, 0.1, size=tmp_Xb.shape)  # Adding random noise as an example
                    new_Xa = torch.tensor(tmp_Xa, dtype=torch.float32)
                    new_Xb = torch.tensor(tmp_Xb, dtype=torch.float32)                  
                    
                new_trn_yy = torch.full((1,),j)
                data_balancing_Xa = torch.cat((data_balancing_Xa,new_Xa), dim=0)
                data_balancing_Xb = torch.cat((data_balancing_Xb,new_Xb), dim=0)
                trn_yy = torch.cat((trn_yy, new_trn_yy),dim=0)
            
        # concat tensor with 10 classes
        if Xa == None:
            if data_balancing_Xa != None:
                Xa = data_balancing_Xa
                Xb = data_balancing_Xb
                trn_y = trn_yy
        else:
            if data_balancing_Xa != None:
                Xa = torch.cat((Xa, data_balancing_Xa), dim=0)
                Xb = torch.cat((Xb, data_balancing_Xb), dim=0)
                trn_y = torch.cat((trn_y, trn_yy), dim=0)
            
    trn_X = [Xa, Xb]
    return trn_X, trn_y


