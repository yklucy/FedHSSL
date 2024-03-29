import os
import argparse
import logging
import sys
import time
import torch

# experiment arguments
def prepare_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NUSWIDE', help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--dataset_dir', type=str, default='/Volumes/Summer_2023/Python/CSI6900_vfl/', help='dataset directory')
    parser.add_argument('--experiment_dir', default='experiment', help='directory for saving experiment logs')
    parser.add_argument('--mul_classes', type = list, default= ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake'], help='10 classes of NUSWIDE')
    parser.add_argument('--balanced', type=int, default=1, help='1: balanced, 0: imbalanced')
    # classic
    parser.add_argument('--pretrained_path', type=str, default='', help='part of directory for saving models')
    parser.add_argument('--freeze_backbone', type=int, default=0, help='0: no freeze; 1: freeze all; 2: freeze passive')
    parser.add_argument('--use_local_model', type=int, default=1, help='whether to use local model')
    parser.add_argument('--use_cross_model', type=int, default=1, help='whether to use cross model')
    parser.add_argument('--pool', type=str, default='mean', help='pooling method for vfl classification task')
    parser.add_argument('--num_cls_layer', type=int, default=1, help='layers of the classification head')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    
    parser.add_argument('--k', type=int, default=2, help='number of clients')
    
    parser.add_argument('--labeled_samples_percent', type=float, default=1.0, help='percentage of aligned samples with labels')
    parser.add_argument('--aligned_samples_percent', type=float, default=0.2, help='percentage of aligned samples')
    parser.add_argument('--valid_percent', type=float, default=0.0)
    
    parser.add_argument('--cls_iso_sigma', type=float, default=0.0, help='coef for mutual training from local to cross')
    parser.add_argument('--cls_iso_threshold', type=float, default=5.0, help='constraint on local model output')
    parser.add_argument('--pt_feat_iso_sigma', type=float, default=0.0, help='defense strength of feature in pretraining phase')
    
    parser.add_argument('--cls_model_dir', default='clsmodels', help='clsmodels')
    
    # training
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    parser.add_argument('--num_output_ftrs', type=int, default=512, help='feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim of mlp encoder')
    
    # pretrain ssl
    parser.add_argument('--local_ssl', type=int, default=0, help='0: disable local ssl; 1: enable local ssl')
    parser.add_argument('--pretrain_model_dir', default='premodels', help='save dir for pretrained model')
    parser.add_argument('--local_epochs_local', type=int, default=1)
    parser.add_argument('--local_epochs_cross', type=int, default=1)
    
    # pretrain
    parser.add_argument('--out_dim', type=int, default=512, help='out dim of head')
    parser.add_argument('--proj_hidden_dim', type=int, default=512, help='proj_hidden_dim')
    parser.add_argument('--pred_hidden_dim', type=int, default=128, help='pred_hidden_dim')
    parser.add_argument('--proj_layer', type=int, default=3, help='projector layer')
    parser.add_argument('--constraint_ratio', type=float, default=0.0, help='constraint on local model output')
    parser.add_argument('--local_ratio', type=float, default=0.5, help='learning rate for ')
    parser.add_argument('--pretrain_lr_head', type=float, default=0.05)
    parser.add_argument('--pretrain_lr_encoder', type=float, default=0.05)
    parser.add_argument('--pretrain_lr_decay', type=int, default=1, help='0: constant; 1: cosine decay ,except for predictor')
    parser.add_argument('--aggregation_mode', type=str, default='none', help='none: no agg; pma: partial agg')
    
    args = parser.parse_args()
    
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    
    return args

def save_exp_logs(args, model_type):
    # save experiment logs
    exp_filename = '{}/{}-{}-{}-{}/'.format(args.experiment_dir, args.dataset, model_type, 'exp', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(exp_filename,exist_ok=True)
    
    # set up log format
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(exp_filename, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    print('Saving expermental logs to: {}'.format(exp_filename))