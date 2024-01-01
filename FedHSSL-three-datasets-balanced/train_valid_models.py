import torch
from fedhssl_models import encrypt_with_iso

#  that is the core of train .....
def step_cross_model(data_loader, client_models_list, epoch, args, step_mode='train', debug=False):
    print('step_cross_model - - -  this is the core of train and validation')
    k = len(client_models_list)
    losses = AverageMeter()
    loss_per_client = [[] for i in range(k)]

    for step, (data_X, data_Y) in enumerate(data_loader):
        data_X = [data_X[idx] for idx in range(args.k)]

        if isinstance(data_X[0], dict) or isinstance(data_X[0], list):
            print('data_X[0] is dict or list!!!!!!!!!!!!!!')
            pass
        else:
            data_X = [x.float().to(args.device) for x in data_X]


        target = data_Y.view(-1).long().to(args.device)
        N = target.size(0)

        exchanged_features = [client_models_list[i].get_exchanged_feature(data_X[i]) for i in range(k)]
        exchanged_features_for_transfer = [feature.detach().clone() for feature in exchanged_features]
        # default pt_feat_iso_sigma = 0
        if args.pt_feat_iso_sigma > 0:
            with torch.no_grad():
                exchanged_features_for_transfer[0] = encrypt_with_iso(exchanged_features_for_transfer[0],
                                                                              args.pt_feat_iso_sigma,
                                                                              args.pt_iso_threshold,
                                                                              args.device)

        # get loss
        loss_total = 0
        for i, client_models in enumerate(client_models_list):
            if i == 0:
                # for the active party (the party has label).
                # The active party receives features from all other parties.
                if step_mode == 'train':
                    # client_models.train_cross_model_loss(self, x, y, z_cross_own, z_cross_received, epoch)
                    loss, cross_meta = client_models.train_cross_model_loss(data_X[i], target, exchanged_features[i],
                                                                exchanged_features_for_transfer[1:], epoch)
                else:
                    loss, cross_meta = client_models.valid_cross_model_loss(data_X[i], target, exchanged_features[i],
                                                                exchanged_features_for_transfer[1:], epoch)
            else:
                # for the passive party (the party has no label).
                # The passive party receives features only from the active party
                if step_mode == 'train':
                    loss, cross_meta = client_models.train_cross_model_loss(data_X[i], None, exchanged_features[i],
                                                                exchanged_features_for_transfer[0], epoch)
                else:
                    loss, cross_meta = client_models.valid_cross_model_loss(data_X[i], None, exchanged_features[i],
                                                                exchanged_features_for_transfer[0], epoch)
            loss_total = loss_total + loss
            loss_per_client[i].append(loss)

        losses.update(loss_total / k, N)
    loss_per_client = [sum(item) / len(item) for item in loss_per_client]
    meta = {'loss_per_client': loss_per_client}
    return losses.avg, meta


def step_local_model(data_loader, client_models_list, epoch, args, step_mode='train'):
    k = len(client_models_list)

    losses = AverageMeter()
    loss_per_client = None
    loss_per_client_reg = None
    for local_epoch in range(args.local_epochs_local):
        # only record the last local epoch
        loss_per_client = [[] for i in range(k)]
        loss_per_client_reg = [[] for i in range(k)]

        for step, (data_X, data_Y) in enumerate(data_loader):
            data_X = [data_X[idx] for idx in range(args.k)]

            if isinstance(data_X[0], dict) or isinstance(data_X[0], list):
                pass
            else:
                data_X = [x.float().to(args.device) for x in data_X]
            target = data_Y.view(-1).long().to(args.device)

            N = target.size(0)
            loss_total = 0
            # local SSL
            for i, client_models in enumerate(client_models_list):
                if step_mode == 'train':
                    loss, local_meta = client_models.train_local_model_loss(data_X[i], target, epoch)
                else:
                    loss, local_meta = client_models.valid_local_model_loss(data_X[i], target, epoch)
                loss_per_client_reg[i].append(local_meta['loss_debug'])

                loss_per_client[i].append(loss)
                loss_total = loss_total + loss
            if local_epoch == args.local_epochs_local - 1:
                losses.update(loss_total / k, N)
        loss_per_client = [sum(item) / len(item) for item in loss_per_client]
        loss_per_client_reg = [sum(item) / len(item) for item in loss_per_client_reg]

    meta = {'loss_per_client': loss_per_client, 'loss_per_client_reg': loss_per_client_reg}
    return losses.avg, meta

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