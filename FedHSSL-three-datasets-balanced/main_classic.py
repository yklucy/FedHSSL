import torch
import copy
import numpy as np
import logging
import random

import torch.utils
import torch.nn as nn

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from exp_arguments import prepare_exp, save_exp_logs
from data.nuswide import load_dataset_nuswide
from data.cifar10 import load_dataset_cifar10, get_meta, get_param_cifar10
from data.emnist import load_dataset_emnist
from fedhssl_models import get_encoder_models_local_bottom, get_encoder_models_local_top, get_encoder_models_cross
from fedhssl_models import ClassificationModelHost, ClassificationModelGuest
from fedhssl_models import encrypt_with_iso
from collections import Counter
from batch_balanced import balance_with_labels


""" Classical VFL: Host + Guest """
def main():
    # read args
    args = prepare_exp()
    args.local_ssl = 1

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    #args.dataset = 'CIFAR10'
    #args.dataset = 'EMNIST'
    print('args.dataset:{}'.format(args.dataset))
    
    #args.balanced = 0 -> imbalanced; default -> 1
    args.balanced = 0

    # save logs
    save_exp_logs(args, 'classic')
    
    """ Step 1: load datasets """
    if args.dataset == 'NUSWIDE':
        print('hi........it is NUSWIDE')
        train_aligned_loader, train_local_loader, test_loader, args = load_dataset_nuswide(args, None)
        mul_classes = args.mul_classes
        num_classes= len(mul_classes)
        input_dims = [634, 1000]
            
    elif args.dataset == 'CIFAR10':
        print('hi........it is CIFAR10')
        args.mul_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        train_aligned_loader, train_local_loader, test_loader, args = load_dataset_cifar10(args, None)
        num_classes= len(args.mul_classes)
        input_dims = [1536, 1536]
    elif args.dataset == 'EMNIST':
        print('hi........it is EMNIST')
        input_dims = [392, 392]
        train_aligned_loader, train_local_loader, test_loader, args = load_dataset_emnist(args, None)  
        num_classes = len(args.mul_classes)
        print('classes len:{}'.format(num_classes))

    """ Step 2: get models"""
    encoder_models_local_bottom = get_encoder_models_local_bottom(args, input_dims)
    encoder_models_local_top = get_encoder_models_local_top(args)
    encoder_models_cross = get_encoder_models_cross(args, input_dims)
    
    """ Step 3: get z """
    use_local_model = True
    use_cross_model = False
    
    # if use pretrained model and use local data and flag of use_local_model is set
    # default args.use_local_model = 1; default pretrained_path = ''; default local_ssl = 0
    # classic method only uses cross models
    if args.use_local_model and args.pretrained_path != '' and args.local_ssl != 0:
        use_local_model = True

    # default args.use_cross_model = 1
    if args.use_cross_model:
        use_cross_model = True
    
    # classic: models used: cross True, local False
    logging.info("models used: cross {}, local {}".format(use_cross_model, use_local_model))
    print('classes len:{}'.format(num_classes))
    # store main model and guest model(s) - model_list[]
    model_list = []
    clsmodel_main = ClassificationModelHost(copy.deepcopy(encoder_models_local_bottom[0]),
                                                 copy.deepcopy(encoder_models_local_top[0]),
                                                 copy.deepcopy(encoder_models_cross[0]),
                                                 args.num_output_ftrs * args.k, num_classes,
                                                 use_cross_model, use_local_model, args.pool, 0.5, args.num_cls_layer)
    model_list.append(clsmodel_main)
    for i in range(args.k - 1):
        guest_model = ClassificationModelGuest(copy.deepcopy(encoder_models_local_bottom[i + 1]),
                                               copy.deepcopy(encoder_models_local_top[i + 1]),
                                               copy.deepcopy(encoder_models_cross[i + 1]),
                                               use_cross_model, use_local_model, args.pool)
        model_list.append(guest_model)
        
    
    """ Step 4: model to device """
    model_list = [model.to(args.device) for model in model_list]

    """ Step 5: load saved models """
    if args.pretrained_path != '':
        print('@@@@@@@@@@@@ load saved models: encoder cross, local bottom, local top @@@@@@@@@@@@@')
        print('args.pretrained_path:{}'.format(args.pretrained_path))
        
        for i in range(args.k):
            if use_cross_model:
                model_list[i].load_encoder_cross(
                    './{}/{}/model_encoder_cross-'.format(args.pretrain_model_dir, args.dataset) + args.pretrained_path + '-{}.pth'.format(i),
                    args.device)
            if use_local_model:
                model_list[i].load_encoder_local_bottom('./{}/{}/model_encoder_local_bottom-'.format(args.pretrain_model_dir, args.dataset) +
                                             args.pretrained_path + '-{}.pth'.format(i), args.device)
                model_list[i].load_encoder_local_top(
                    './{}/{}/model_encoder_local_top-'.format(args.pretrain_model_dir, args.dataset)+ args.pretrained_path + '-{}.pth'.format(i),
                    args.device)

        logging.info('***** USE PRETRAIN MODEL： {}, {}, {}'.format(args.pretrain_model_dir, args.dataset, args.pretrained_path))
    

    
    """ Step 6: Freeze backbone """
    # default freeze_backbone = 0
    if args.freeze_backbone == 1:
        # all model backbone freezed
        for model in model_list:
            model.freeze_backbone()
        logging.info('***** FREEZE BACKBONE')
    elif args.freeze_backbone == 2:
        # first model is active, encoder_cross is active
        for model in model_list[1:]:
            model.freeze_backbone()
        logging.info('***** FREEZE BACKBONE, EXCEPT THE FIRST')

        
    """ Step 7: set criterion, weights optimizer """
    # criterion
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    # weights optimizer
    optimizer_list = [torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay) for model in model_list]
    
    scheduler_list = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) for optimizer in optimizer_list]
    best_acc_top1 = 0.
    best_f1_score_micro = 0.
    model_f1_score_micro = 0.
    flag_model = False
    
    """ Step 8: train and save models"""
    for epoch in range(args.epochs):
        print('hi.....I am training....{} times'.format(epoch))
        train_losses, train_acc, train_f1_score_micro, train_precision, train_recall, train_f1, train_roc_auc = train(train_aligned_loader, model_list, optimizer_list, criterion, epoch, args)

        # print avg train results
        for i in range(num_classes):
            if i in train_precision:
                logging.info(f'Class {i} - Precision: {train_precision[i]:.2f}, Recall: {train_recall[i]:.2f}, F1-Score: {train_f1[i]:.2f}, ROC_AUC: {train_roc_auc[i]:.2f}')
            else:
                logging.info(f'Class {i} - Precision: 0.00, Recall: 0.00, F1-Score: 0.00, ROC_AUC: {train_roc_auc[i]:.2f}')
        
        '''
        for i in range(num_classes):
            if (i in train_precision) and (i in train_recall) and (i in train_f1) and (i in train_roc_auc):
                logging.info(f'Class {i} - Precision: {train_precision[i]:.10f}, Recall: {train_recall[i]:.10f}, F1-Score: {train_f1[i]:.10f}, ROC_AUC: {train_roc_auc[i]:.2f}')
            elif i in train_precision:
                logging.info(f'Class {i} - Precision: {train_precision[i]:.10f}')
            elif i in train_recall:
                logging.info(f'Class {i} - Precision: None, Recall: {train_recall[i]:.10f}')
            elif i in train_f1:
                logging.info(f'Class {i} - Precision: None, Recall: None, F1-Score: {train_f1[i]:.10f}')
            elif i in train_roc_auc:
                logging.info(f'Class {i} - Precision: None, Recall: None, F1-Score: None, ROC_AUC: {train_roc_auc[i]:.2f}')
            else:
                logging.info(f'No Class {i} in train dataset!')
        '''
             
        # validation / test
        cur_step = (epoch+1) * len(train_aligned_loader)
        test_losses, test_acc, test_f1_score_micro, test_precision, test_recall, test_f1, test_roc_auc = validate(test_loader, model_list, criterion, epoch, cur_step, args)
        # print avg test results
        for i in range(num_classes):
            if i in test_precision:
                logging.info(f'Class {i} - Precision: {test_precision[i]:.2f}, Recall: {test_recall[i]:.2f}, F1-Score: {test_f1[i]:.2f}, ROC_AUC: {test_roc_auc[i]:.2f}')
            else:
                logging.info(f'Class {i} - Precision: 0.00, Recall: 0.00, F1-Score: 0.00, ROC_AUC: {test_roc_auc[i]:.2f}')
        
        # save
        if test_acc > best_acc_top1:
            best_acc_top1 = test_acc
        logging.info('best_acc_top1 %f', best_acc_top1)

        if test_f1_score_micro > best_f1_score_micro:
            best_f1_score_micro = test_f1_score_micro
        logging.info('best_f1_score_micro %f', best_f1_score_micro)

        for scheduler in scheduler_list:
            scheduler.step()
    
    """ Step 9: save models from passie client - ClassificationModelGuest"""
    name_str = 'mlp2'
    model_list[-1].save_models(args.cls_model_dir, args.dataset, name_str)
    
    print('model_list[-1]:{}'.format(model_list[-1]))
    logging.info("***** model saved *****")
    print("***** save mode ****")


""" functions """

def train(train_loader, model_list, optimizer_list, criterion, epoch, args):
    acc_train = AverageMeter()
    
    f1_score_micro_train = AverageMeter()
    
    losses = AverageMeter()
    num_classes = len(args.mul_classes)
    precision_train = AverageMeterDict(num_classes)
    recall_train = AverageMeterDict(num_classes)
    f1_train = AverageMeterDict(num_classes)
    roc_auc_train = AverageMeterDict(num_classes)

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer_list[0].param_groups[0]['lr']
    # Epoch 0, cur_step 0, LR 0.025
    logging.info("Epoch {}, cur_step {}, LR {}".format(epoch, cur_step, cur_lr))
    
    # set all models to train()
    for model in model_list:
        model.train()

    k = len(model_list)

    # idx = 0, 1
    # get data by batch
    for step, (trn_X_in, trn_y_in) in enumerate(train_loader):
        trn_X_in = [trn_X_in[idx] for idx in range(k)]
        
        # dict for bert model
        if isinstance(trn_X_in[0], dict):
            pass
        else:
            trn_X_in = [x.float().to(args.device) for x in trn_X_in]
        
        if args.balanced == 1:
            print('Now, balancing.....')
            num = 50
            trn_X, trn_y = balance_with_labels(trn_X_in, trn_y_in, num, args)
        else:
            trn_X, trn_y = trn_X_in, trn_y_in
            
        target = trn_y.view(-1).long().to(args.device)
        N = target.size(0)
        z_rest_clone = None

        """ z = f(backbone(x)) / no projection"""
        z_list = [model_list[i](trn_X[i]) for i in range(0, len(model_list))]
        z_0 = z_list[0]
        if k > 1:
            z_rest = z_list[1:]
            z_rest_clone = [z.detach().clone() for z in z_rest]
            z_rest_clone = [torch.autograd.Variable(z, requires_grad=True).to(args.device) for z in z_rest_clone]

        """ p = prediction(z) / classifier_head() """
        logits = model_list[0].get_prediction(z_0, z_rest_clone)  
        loss = criterion(logits, target)
        
        # for guest client(s)
        if k > 1:
            if args.freeze_backbone == 0:
                z_gradients_list = [torch.autograd.grad(loss, z, retain_graph=True) for z in z_rest_clone]
                if args.cls_iso_sigma > 0:
                    z_gradients_list = [encrypt_with_iso(z[0], args.cls_iso_sigma, args.cls_iso_threshold, args.device)
                                              for z in z_gradients_list]

                weights_gradients_list = [
                    torch.autograd.grad(z_rest[i], model_list[i + 1].parameters(), grad_outputs=z_gradients_list[i],
                                        retain_graph=True) for i in range(len(z_gradients_list))] 
        optimizer_list[0].zero_grad()
        
        loss.backward()  # retain_graph=True
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model_list[0].parameters(), args.grad_clip)
        optimizer_list[0].step()

        # for Guest client(s)
        if k > 1:
            if args.freeze_backbone == 0:
                [optimizer_list[i].zero_grad() for i in range(1, k)]
                for i in range(len(weights_gradients_list)):
                    for w, g in zip(model_list[i + 1].parameters(), weights_gradients_list[i]):
                        if w.requires_grad:
                            w.grad = g.detach()
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model_list[i + 1].parameters(), args.grad_clip)
                    optimizer_list[i + 1].step()


        acc, f1_score_micro, precision_dict, recall_dict, f1_dict, roc_auc_dict = perf_metrics(logits, target, args)
            
        losses.update(loss.item(), N)
        acc_train.update(acc.item(), N)

        f1_score_micro_train.update(f1_score_micro.item(), N)

        precision_train.update(precision_dict, N)
        recall_train.update(recall_dict, N)
        f1_train.update(f1_dict, N)
        roc_auc_train.update(roc_auc_dict, N)
        
        if step % args.report_freq == 0 or step == len(train_loader) - 1:
            logging.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:6f}%)".format(
                    epoch + 1, args.epochs, step, len(train_loader) - 1, losses=losses, top1=acc_train))

        cur_step += 1

    return losses.avg, acc_train.avg, f1_score_micro_train.avg, precision_train.avg, recall_train.avg, f1_train.avg, roc_auc_train.avg

def validate(valid_loader, model_list, criterion, epoch, cur_step, args):
    acc_test = AverageMeter()
    f1_score_micro_test = AverageMeter()

    losses = AverageMeter()
    num_classes = len(args.mul_classes)
    precision_test = AverageMeterDict(num_classes)
    recall_test = AverageMeterDict(num_classes)
    f1_test = AverageMeterDict(num_classes)
    roc_auc_test = AverageMeterDict(num_classes)
    
    # set model to validation mode
    for model in model_list:
        model.eval()

    k = len(model_list)

    with torch.no_grad():
        for step, (val_X, val_y) in enumerate(valid_loader):
            val_X = [val_X[idx] for idx in range(k)]
            if isinstance(val_X[0], dict):
                pass
            else:
                val_X = [x.float().to(args.device) for x in val_X]
            target = val_y.view(-1).long().to(args.device)
            N = target.size(0)

            z_rest_clone = None

            z_list = [model_list[i](val_X[i]) for i in range(0, len(model_list))]
            z_0 = z_list[0]
            if k > 1:
                z_rest = z_list[1:]
                z_rest_clone = [z.detach().clone() for z in z_rest]
                z_rest_clone = [torch.autograd.Variable(z, requires_grad=True).to(args.device) for z in
                                z_rest_clone]

            logits = model_list[0].get_prediction(z_0, z_rest_clone)

            loss = criterion(logits, target)
            acc, f1_score_micro, precision_dict, recall_dict, f1_dict, roc_auc_dict = perf_metrics(logits, target, args)

            losses.update(loss.item(), N)
            acc_test.update(acc.item(), N)
            f1_score_micro_test.update(f1_score_micro.item(), N)

            precision_test.update(precision_dict, N)
            recall_test.update(recall_dict, N)
            f1_test.update(f1_dict, N)
            roc_auc_test.update(roc_auc_dict, N)


    # print/logging acc for datasets
    logging.info(
        "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
        "Prec@(1,5) ({top1.avg:.6f}%)".format(
            epoch + 1, args.epochs, step, len(valid_loader) - 1, losses=losses, top1=acc_test))
    return losses.avg, acc_test.avg, f1_score_micro_test.avg, precision_test.avg, recall_test.avg, f1_test.avg, roc_auc_test.avg



"""@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""
""" Other Classes and functions """

class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class AverageMeterDict():
    def __init__(self, len):
        self.len = len
        self.reset()
    
    def reset(self):
        self.sum = {}
        self.count = {}
        self.avg = {}
    # by batch                    
    def update(self, val_dict, n=1):
        self.val = {}
        for key, value in val_dict.items():
            self.val[key] = value
            if key in self.sum:
                self.sum[key] = self.sum[key] + self.val[key] * n
            else:
                self.sum[key] = self.val[key] * n
            if key in self.count:
                self.count[key] = self.count[key] + n
            else:
                self.count[key] = n
            self.avg[key] = self.sum[key] / self.count[key]
          
def perf_metrics(logits, target, args):
    batch_size = target.size(0)

    _, pred = logits.topk(1, 1, True, True)

    pred_t = pred.t()
    correct = pred_t.eq(target.view(1, -1).expand_as(pred_t))
    correct_k = correct[0].reshape(-1).float().sum(0)
    
    acc = correct_k / batch_size
    
    
    ''' Precision, recall, f1-score and roc_auc '''
    pred_t_list = pred_t[0]
    target_au = np.array(target.tolist())
    pred_t_list_au = np.array(pred_t_list.tolist())
    # 10
    num_classes = len(args.mul_classes)
    # <= 10
    unique_classes = np.unique(target)
    
    f1_score_micro = f1_score(target_au, pred_t_list_au, average='micro', zero_division=0.0)

    # array
    precision = precision_score(target_au, pred_t_list_au, average=None, zero_division=0.0)
    recall = recall_score(target_au, pred_t_list_au, average=None, zero_division=0.0)
    f1 = f1_score(target_au, pred_t_list_au, average=None, zero_division=0.0)

    # initial dict with 10
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    # store values as dict
    for i in range(len(precision)):
        if precision[i] != 0.0:
            precision_dict[i] = precision[i]
    for i in range(len(recall)):
        if recall[i] != 0.0:
            recall_dict[i] = recall[i]
    for i in range(len(f1)):
        if f1[i] != 0.0:
            f1_dict[i] = f1[i]
    
    
    # Calculate ROC AUC score for each class
    roc_auc_dict = {}
    for class_label in unique_classes:
        ovr_target = [1 if label == class_label else 0 for label in target]
        ovr_pred = [1 if pred == class_label else 0 for pred in pred_t_list]
        roc_auc = roc_auc_score(ovr_target, ovr_pred)
        roc_auc_dict[class_label] = roc_auc

    return acc, f1_score_micro, precision_dict, recall_dict, f1_dict, roc_auc_dict

            
if __name__ == '__main__':
    main()