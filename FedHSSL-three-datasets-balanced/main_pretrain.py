import random
import torch
import numpy as np
import logging

from exp_arguments import prepare_exp, save_exp_logs
from data.nuswide import load_dataset_nuswide
from data.cifar10 import load_dataset_cifar10
from data.emnist import load_dataset_emnist
from fedhssl_models import get_encoder_models_local_bottom, get_encoder_models_local_top, get_encoder_models_cross
from fedhssl_models import Clients_pretrain_models, SSServer_aggregate
from train_valid_models import step_cross_model, step_local_model

def main():
    ''' load args '''
    print('Step 1: loading args....')
    args = prepare_exp()
    
    # local_ssl = 1 -> enable local_ssl
    args.local_ssl = 1
    args.aggregation_mode = 'pma'
    print('args:{}'.format(args))
    
    #args.dataset = 'CIFAR10'
    #args.dataset = 'EMNIST'
    print('args.dataset:{}'.format(args.dataset))
    
    # save logs
    save_exp_logs(args, 'pretrain')
    
    """ set seed """
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    
    """ Step 2: load datasets """
    if args.dataset == 'NUSWIDE':
        print('hi..........it is NUSWIDE')
        train_aligned_loader, train_local_loader, test_loader, args = load_dataset_nuswide(args, 'pretrain')
        # 10 classes
        #mul_classes = ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']
        mul_classes = args.mul_classes
        num_classes= len(mul_classes) 
        input_dims = [634, 1000]
    elif args.dataset == 'CIFAR10':
        print('hi........it is CIFAR10')
        args.mul_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        train_aligned_loader, train_local_loader, test_loader, args = load_dataset_cifar10(args, 'pretrain')
        num_classes= len(args.mul_classes)
        input_dims = [1536, 1536]
    elif args.dataset == 'EMNIST':
        print('hi........it is EMNIST')
        input_dims = [392, 392]
        train_aligned_loader, train_local_loader, test_loader, args = load_dataset_emnist(args, 'pretrain')       
        num_classes = len(args.mul_classes)
        print('classes len:{}'.format(num_classes))
        
    print('train_loader_aligned len{}'.format(len(train_aligned_loader)))
        
    """ Step 3: get backbone models"""
    encoder_models_local_bottom = get_encoder_models_local_bottom(args, input_dims)
    encoder_models_local_top = get_encoder_models_local_top(args)
    encoder_models_cross = get_encoder_models_cross(args, input_dims)

    #save args to log
    logging.info("args = {}".format(vars(args)))
    
    
    """ Step 4: get clients' models -> client_models_list[0,1] """
    client_models_list = []
    # i = 0, 1
    for i in range(args.k):
        client_models = Clients_pretrain_models(i, [encoder_models_local_bottom[i], encoder_models_local_top[i], encoder_models_cross[i]], args)
        client_models_list.append(client_models)

    print('client_models_list:{}'.format(client_models_list))
    

    """ Step 5: Initial a Server for aggregating local models' state_dict  """
    Server = SSServer_aggregate(args)
    print("Server:{}".format(Server))
    
    
    """ Step 6: start training and validation """
    """ start training """
    best_valid_loss = None
    best_epoch = 0
    for epoch in range(args.epochs):
        print('epoch {}:'.format(epoch))
        # ----- train ----- #
        """ Part 1: train cross model """
        print('* * * * * train cross model * * * * ')
        sample_num = len(train_aligned_loader)
        cur_lr = client_models_list[0].optimizer_list_cross[0].param_groups[0]['lr']
        logging.info("Cross-Party Train Epoch {}, training on aligned data, LR: {}, sample: {}".format(epoch, cur_lr, sample_num * args.batch_size))
        
        for client_models in client_models_list:
            client_models.set_train()
        
        loss, meta = step_cross_model(train_aligned_loader, client_models_list, epoch, args, 'train')
        loss_per_client = meta['loss_per_client']
        
        logging.info("Cross-Party SSL Train Epoch {}, per aligned clients' loss : {}".format(epoch, loss_per_client))
        
        """ Part 2: train local model and aggregating by Server """
        # default local_ssl = 0
        if args.local_ssl != 0:
            print('local_ssl !=0 and * * * * train local model * * * * ')

            sample_num = len(train_local_loader)
            try:
                cur_lr = client_models_list[0].optimizer_list_local[0].param_groups[0]['lr']
            except:
                cur_lr = client_models_list[0].optimizer_list_cross[0].param_groups[0]['lr']

            logging.info("Local SSL Train Epoch {}, training on local data, sample: {}".format(epoch, sample_num * args.batch_size))
    
            for client_models in client_models_list:
                client_models.set_train()

            loss, meta = step_local_model(train_local_loader, client_models_list, epoch, args, 'train')
            loss_per_client = meta['loss_per_client']
            logging.info("Local SSL Train Epoch {}, client loss local: {}".format(epoch, loss_per_client))
            
            # server: aggregation , local: replace
            print("train_local_model / server: aggregation")
            loss_agg = Server.aggregation(client_models_list, sample_num)
            logging.info("Local SSL Train Epoch {}, AGG MODE {}, client loss agg: {}".format(epoch, args.aggregation_mode, loss_agg))
            
        
        
        """ Part 3: adjust learning rate """    
        # default pretrain_lr_decay = 1; len(client_list) = 2
        if args.pretrain_lr_decay == 1:
            print('* * * * * adjust_learning_rate * * * * *')
            for client_models in client_models_list:
                client_models.adjust_learning_rate()
            Server.adjust_learning_rate()


        """ Part 4: Validation / Test """
        # ----- validation ----- #
        # default valid_percent = 0.0
        print('* * * * * valid/test using test_loader * * * * ')
        # set models to validation
        for client_models in client_models_list:
            client_models.set_eval()

        loss_cross = 0
        loss_local = 0
        with torch.no_grad():
            # same valid loader for both cross-party and local SSL
            # valid_load / test_load -> [test_dataset, test_aug_dataset]
            if not isinstance(test_loader, list):
                loss_cross, meta_cross = step_cross_model(test_loader, client_models_list, epoch, args, 'valid')
                # default local_ssl = 0 -> local
                if args.local_ssl != 0:
                    loss_local, meta_local = step_local_model(test_loader, client_models_list, epoch, args, 'valid')
            else:
                loss_cross, meta_cross = step_cross_model(test_loader[0], client_models_list, epoch, args, 'valid')
                if args.local_ssl != 0:
                    loss_local, meta_local = step_local_model(test_loader[1], client_models_list, epoch, args, 'valid')

        logging.info("###### Valid/Test Epoch {} Start #####".format(epoch))
        logging.info("Valid/Test Epoch {}, valid/test loss for per aligned clients: {}".format(epoch, meta_cross['loss_per_client']))

        # default local_ssl != 0 -> enable local
        if args.local_ssl != 0:
            logging.info("Valid/Test Epoch {}, valid/test client loss local: {}".format(epoch, meta_local['loss_per_client']))
            logging.info("Valid/Test Epoch {}, valid/test client loss regularized: {}".format(epoch, meta_local['loss_per_client_reg']))

        logging.info(
            "Valid/Test Epoch {}, Loss_aligned {losses_cross:.3f} Loss_local {losses_local:.3f}".format(
            epoch, losses_cross=loss_cross, losses_local=loss_local))

        logging.info("###### Valid Epoch {} End #####".format(epoch))

        """ Part 5: save best model """
        print('* * * * * save best model * * * * ')
        # save best model
        if best_valid_loss is None:
            best_valid_loss = loss_cross
        else:
            if loss_cross < best_valid_loss:
                best_valid_loss = loss_cross
                best_epoch = epoch
                name_str = 'mlp2'
                for i in range(args.k):
                    client_models_list[i].save_models(args.pretrain_model_dir, args.dataset, name_str, i)
                logging.info("***** best model saved at epoch {} *****".format(epoch))
        logging.info("***** best loss {}: {} *****".format(best_epoch, best_valid_loss))
        
    
    print('Step 7: postprocess traning')
    """ postprocess training """
    # ----- save pretrained models ----- #
    name_str = 'mlp2'
    for i in range(args.k):
        client_models_list[i].save_models(args.pretrain_model_dir, args.dataset, name_str, i)
    logging.info("***** model saved *****")

    # ----- clean up ----- #
    logging.info("***** results logged *****")
    
    
if __name__ == '__main__':
    main()