import os
import torch
import torch.utils
import torch.nn as nn
import copy

from exp_arguments import prepare_exp

args = prepare_exp()

"""model group 1 (backbone): cross_party model, inheritance from nn.Module"""
class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 512]):
        super(MLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def get_encoder_models_cross(args, input_dims):
    encoder_models_cross = []
    num_output_ftrs = args.num_output_ftrs
    hidden_dim = args.hidden_dim
    
    for i in range(args.k):
        encoder_model = MLP2(input_dims[i], [hidden_dim, num_output_ftrs])
        encoder_models_cross.append(encoder_model)
    return encoder_models_cross



"""model group 2 (backbone): local models: BottomMLP2 and TopMLP2, inheritance from nn.Module"""
# inheritance nn.Module
class BottomMLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(BottomMLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x

# inheritance nn.Module
class TopMLP2(nn.Module):
    def __init__(self, hidden_dims=[512, 512]):
        super(TopMLP2, self).__init__()
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer2(x)
        return x


def get_encoder_models_local_bottom(args, input_dims):
    encoder_models_local_bottom = []
    hidden_dim = args.hidden_dim
    
    for i in range(args.k):
        encoder_model_local_bottom = BottomMLP2(input_dims[i], hidden_dim)
        encoder_models_local_bottom.append(encoder_model_local_bottom)
    return encoder_models_local_bottom


def get_encoder_models_local_top(args):
    encoder_models_local_top = []
    num_output_ftrs = args.num_output_ftrs
    hidden_dim = args.hidden_dim
    
    for i in range(args.k):
        encoder_model_local_top = TopMLP2([hidden_dim, num_output_ftrs])
        encoder_models_local_top.append(encoder_model_local_top)
    return encoder_models_local_top


"""model group 3 (Cross-party SSL pretrain models): projection_mlp_cross and prediction_mlp_cross, inheritance from nn.Module"""
def prediction_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int) -> (nn.Sequential):
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims, bias=False),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))
    l2 = nn.Linear(h_dims, out_dims)
    prediction = nn.Sequential(l1, l2)
    return prediction

def projection_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int,
                    num_layers: int = 3) -> (nn.Sequential):
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims, bias=False),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))
    l2 = nn.Sequential(nn.Linear(h_dims, h_dims, bias=False),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))
    l3 = nn.Sequential(nn.Linear(h_dims, out_dims),
                       nn.BatchNorm1d(out_dims))
    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError("Only MLPs with 2 and 3 layers are implemented.")

    return projection


"""model group 4 (clients' SSL pretrain models): projection_mlp_local and prediction_mlp_local, inheritance from nn.Module"""
class Clients_pretrain_models():
    # models: 
    def __init__(self, client_idx, models, args):
        self.client_idx = client_idx
        self.args = args
        self.device = args.device

        # settings for projector and predictor
        # args.out_dim = 512; project_hidden_dim = 512; pred_hidden_dim = 128; proj_layer = 3
        self.out_dim = args.out_dim
        self.proj_hidden_dim = args.proj_hidden_dim
        self.pred_hidden_dim = args.pred_hidden_dim
        self.num_mlp_layers = args.proj_layer

        # main models and optimizers
        self.encoder_local_bottom = copy.deepcopy(models[0]).to(args.device)
        self.encoder_local_top = copy.deepcopy(models[1]).to(args.device)
        self.encoder_cross = copy.deepcopy(models[2]).to(args.device)
        
        self.projection_mlp_cross = projection_mlp(args.num_output_ftrs, self.proj_hidden_dim, self.out_dim, self.num_mlp_layers).to(args.device)
        self.prediction_mlp_cross = prediction_mlp(self.out_dim, self.pred_hidden_dim, self.out_dim).to(args.device)
        self.projection_mlp_local = projection_mlp(args.num_output_ftrs, self.proj_hidden_dim, self.out_dim, self.num_mlp_layers).to(args.device)
        self.prediction_mlp_local = prediction_mlp(self.out_dim, self.pred_hidden_dim, self.out_dim).to(args.device)
        
        # models for aggregation
        # encoder_local_top = encoder_local
        self.model_local_top = nn.ModuleList([self.encoder_local_top, self.projection_mlp_local, self.prediction_mlp_local])
        
        # models for batch operation
        self.models = nn.ModuleList([self.encoder_cross, self.projection_mlp_cross, self.prediction_mlp_cross,
                                     self.encoder_local_bottom, self.encoder_local_top, self.projection_mlp_local,
                                     self.prediction_mlp_local])
        
        
        # loss
        self.cross_criterion = NegCosineSimilarityLoss().to(args.device)
        self.local_criterion = NegCosineSimilarityLoss().to(args.device)

        # rescale learning rate; args.local_ratio = 0.5; constraint_ratio = 0.0
        self.pretrain_lr_ratio = 0.5/(self.args.local_ratio + self.args.constraint_ratio)
                              
        # optimizer list; optimizer.step() -> updated the parameters
        self.optimizer_list_cross = [self.get_optimizer(self.encoder_cross, self.args.optimizer),
                                     self.get_optimizer(self.projection_mlp_cross),
                                     self.get_optimizer(self.prediction_mlp_cross)]


        self.optimizer_list_local = [self.get_optimizer(self.encoder_local_bottom, self.args.optimizer),
                                     self.get_optimizer(self.encoder_local_top, self.args.optimizer),
                                     self.get_optimizer(self.projection_mlp_local),
                                     self.get_optimizer(self.prediction_mlp_local)]

        # learning rate scheduler - adjust the learning rate based on the number of epochs; scheduler.step()
        self.scheduler_list = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
                                  for optimizer in self.optimizer_list_cross[:-1] if optimizer is not None] + \
                              [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
                               for optimizer in self.optimizer_list_local[:-1] if optimizer is not None]
                              
        self.model_to_device(args.device)


    def model_to_device(self, device):
        for model in self.models:
            model.to(device)
    
    def set_train(self):
        for model in self.models:
            model.train()
    
    def set_eval(self):
        for model in self.models:
            model.eval()
    
    def get_exchanged_feature(self, x):
        if isinstance(x, list):
            h_cross = self.encoder_cross(x[0].float().to(self.args.device)).flatten(start_dim=1)
            z_cross = self.projection_mlp_cross(h_cross)
        else:
            h_cross = self.encoder_cross(x).flatten(start_dim=1)
            z_cross = self.projection_mlp_cross(h_cross)
            
        return z_cross
    
    def adjust_learning_rate(self):
        for scheduler in self.scheduler_list:
            scheduler.step()
    
    
    def get_optimizer(self, model, opt_type='sgd'):
        # default batch_size = 512; default pre_train_lr_head = 0.05; default pre_train_lr_encoder = 0.05
        # default momentum = 0.9; default weight_decay = 3e-05
        # input - train_cross:  model - self.encoder_cross; self.projection_mlp_cross; self.prediction_mlp_cross
        pretrain_lr_head = self.args.pretrain_lr_head * self.args.batch_size / 256
        pretrain_lr_encoder = self.args.pretrain_lr_encoder * self.args.batch_size / 256
        if opt_type == 'sgd':
            return torch.optim.SGD(model.parameters(), pretrain_lr_head * self.pretrain_lr_ratio,
                                   momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif opt_type == 'adagrad':
            return torch.optim.Adagrad(model.parameters(), pretrain_lr_encoder * self.pretrain_lr_ratio)
        else:
            return None

    def opt_preprocess(self, submodel='cross'):
        if submodel == 'cross':
            for opt in self.optimizer_list_cross:
                opt.zero_grad()
        elif submodel == 'local':
            for opt in self.optimizer_list_local:
                opt.zero_grad()
    
    def opt_postprocess(self, submodel='cross'):
        if submodel == 'cross':
            for opt in self.optimizer_list_cross:
                opt.step()
        elif submodel == 'local':
            for opt in self.optimizer_list_local:
                opt.step()
    
    def compute_cross_loss(self, x, y, z_cross_own, z_cross_received, epoch):
        meta = {}
        p_cross_own = self.prediction_mlp_cross(z_cross_own)
        loss_debug = []
        if isinstance(z_cross_received, list):
            loss = torch.tensor(0)
            for z_item in z_cross_received:
                loss_ind = self.cross_criterion(p_cross_own, z_item.detach())
                loss_debug.append(loss_ind.item())
                loss = loss + loss_ind
            loss = loss / len(z_cross_received)
        else:
            loss = self.cross_criterion(p_cross_own, z_cross_received.detach())
        meta['loss_debug'] = loss_debug
        return loss, meta
    
    def compute_local_loss(self, x, y, epoch=0, debug=False):
        meta = {}
        loss_constraint = torch.tensor(0).to(self.args.device)

        x1 = x[0].float().to(self.args.device)
        x2 = x[1].float().to(self.args.device)

        # model outputs of two augemented views
        f1 = self.encoder_local_bottom(x1)
        h1 = self.encoder_local_top(f1).flatten(start_dim=1)
        z1 = self.projection_mlp_local(h1)
        p1 = self.prediction_mlp_local(z1)

        f2 = self.encoder_local_bottom(x2)
        h2 = self.encoder_local_top(f2).flatten(start_dim=1)
        z2 = self.projection_mlp_local(h2)
        p2 = self.prediction_mlp_local(z2)

        # local contrastive loss using augmented views
        loss = self.args.local_ratio * (self.local_criterion(p1, z2.detach()) + self.local_criterion(p2, z1.detach()))

        # regularization loss of cross encoder
        if self.args.constraint_ratio > 0:
            h1_cross = self.encoder_cross(x1).flatten(start_dim=1)
            h2_cross = self.encoder_cross(x2).flatten(start_dim=1)
            z1_cross = self.projection_mlp_cross(h1_cross)
            z2_cross = self.projection_mlp_cross(h2_cross)

            loss_constraint = self.args.constraint_ratio * (self.local_criterion(p1, z1_cross.detach()) +
                                                            self.local_criterion(p2, z2_cross.detach()))
            loss = loss + loss_constraint
        meta['loss_debug'] = loss_constraint.item()
        return loss, meta
    
    # client / train_cross_model
    def train_cross_model_loss(self, x, y, z_cross_own, z_cross_received, epoch):
        loss_total = []
        # local_epochs_cross = 1; local_epoch = 0
        for local_epoch in range(self.args.local_epochs_cross):
            if local_epoch > 0:
                z_cross_own = self.get_exchanged_feature(x)

            self.opt_preprocess('cross')
            loss, cross_meta = self.compute_cross_loss(x, y, z_cross_own, z_cross_received, epoch)
            loss.backward()
            self.opt_postprocess('cross')

            loss_total.append(loss.item())

        loss_mean = sum(loss_total) / len(loss_total)
        return loss_mean, cross_meta
    
    def train_local_model_loss(self, x, y, epoch):
        self.opt_preprocess('local')
        loss, local_meta = self.compute_local_loss(x, y, epoch)
        loss.backward()
        # gradient clip
        if self.args.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.models.parameters(), self.args.grad_clip)

        self.opt_postprocess('local')

        return loss.item(), local_meta
    
    def update_local_top_model(self, backbone_local_state_dict, x=None, device='cpu'):
        if x is None:
            self.model_local_top.load_state_dict(backbone_local_state_dict)
            self.model_local_top.to(device)
            
        else:
            self.model_local_top.to(device)

    
    def get_local_top_model(self, defense_ratio=0.0):
        if defense_ratio > 0.0:
            ret_model = copy.deepcopy(self.model_local_top)
               
            with torch.no_grad():
                for param in ret_model.parameters():
                    param.data = encrypt_with_iso(param.data, defense_ratio)
            return ret_model.cpu().state_dict()

        return self.model_local_top.cpu().state_dict()
    
    def valid_cross_model_loss(self, x, y, z_cross_own, z_cross_received, epoch):
        loss, cross_meta = self.compute_cross_loss(x, y, z_cross_own, z_cross_received, epoch)
        return loss.item(), cross_meta
    
    def valid_local_model_loss(self, x, y, epoch):
        loss, local_meta = self.compute_local_loss(x, y, epoch)
        return loss.item(), local_meta
    
    # target_dir -> dir; name_str -> model name; idx = 0 or 1
    def save_models(self, target_dir, dataset, name_str, idx):
        target_dir = os.path.join(target_dir, dataset)
        os.makedirs(target_dir, exist_ok=True)
        torch.save(self.encoder_cross.state_dict(),
                   os.path.join(target_dir, 'model_encoder_cross-{}-{}.pth'.format(name_str, idx)))
        if self.args.local_ssl:
            torch.save(self.encoder_local_bottom.state_dict(),
                       os.path.join(target_dir, 'model_encoder_local_bottom-{}-{}.pth'.format(name_str, idx)))

            torch.save(self.encoder_local_top.state_dict(),
                       os.path.join(target_dir, 'model_encoder_local_top-{}-{}.pth'.format(name_str, idx)))

    def get_cross_projection_feature(self, h):
        z_cross = self.projection_mlp_cross(h)
        return z_cross

    def get_cross_prediction_feature(self, z):
        p_cross = self.prediction_mlp_cross(z)
        return p_cross

    def get_local_projection_feature(self, h):
        z_local = self.projection_mlp_local(h)
        return z_local

    def get_local_prediction_feature(self, z):
        p_local = self.prediction_mlp_local(z)
        return p_local

    def get_cross_encoder_feature(self, x):
        if isinstance(x, list):
            h_cross = self.encoder_cross(x[0].float().to(self.args.device)).flatten(start_dim=1)
        else:
            h_cross = self.encoder_cross(x).flatten(start_dim=1)
        return h_cross

    def get_local_encoder_feature(self, x):
        if isinstance(x, list):
            f_local = self.encoder_local_bottom(x[0].float().to(self.args.device))
            h_local = self.encoder_local_top(f_local).flatten(start_dim=1)
        else:
            f_local = self.encoder_local_bottom(x)
            h_local = self.encoder_local_top(f_local).flatten(start_dim=1)
        return h_local


class NegCosineSimilarityLoss(torch.nn.Module):
    """Implementation of the Symmetrized Loss used in the SimSiam[0] paper.
    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566
    """

    def _neg_cosine_simililarity(self, p, z):
        v = - torch.nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()
        return v

    def forward(self,
                p: torch.Tensor,
                z: torch.Tensor):
        """Forward pass through Symmetric Loss.
        """
        loss = self._neg_cosine_simililarity(p, z)

        return loss
    
# encrypt with iso
def encrypt_with_iso(g, ratio, th=5.0, device='cpu'):
    g = g.cpu()

    g_original_shape = g.shape
    g = g.view(g_original_shape[0], -1)

    g_norm = torch.norm(g, dim=1, keepdim=False)
    g_norm = g_norm.view(-1, 1)
    max_norm = torch.max(g_norm)
    gaussian_noise = torch.normal(size=g.shape, mean=0.0,
                                  std=1e-6+ratio * max_norm / torch.sqrt(torch.tensor(g.shape[1], dtype=torch.float32)))
    res = g + gaussian_noise
    res = res.view(g_original_shape).to(device)

    return res


"""model group 5 (model_local_top model for aggregating by Server): TopMLP2, prejection_mlp_local, prediction_mlp_local"""
class SSServer_aggregate(object):
    
    def __init__(self, args):
        self.args = args
        self.scheduler_list = []

    def aggregation(self, client_models_list, sample_num):
        print('aggregation from SSServer')
        k = len(client_models_list)
        loss_debug = []
        if self.args.aggregation_mode == 'pma':
            w_locals = []
            for idx, client_model in enumerate(client_models_list):
                # update dataset
                w = client_model.get_local_top_model()
                if isinstance(sample_num, list):
                    w_locals.append((sample_num[idx], copy.deepcopy(w)))
                else:
                    w_locals.append((sample_num, copy.deepcopy(w)))

            # aggregate local models
            global_model_state_dict = aggregate_fedavg(w_locals)

            # update local model models
            for i in range(len(client_models_list)):
                client_models_list[i].update_local_top_model(global_model_state_dict, None, self.args.device)
        return loss_debug

    def adjust_learning_rate(self):
        print('scheduler.step() to adjust_learning_rate')
        for scheduler in self.scheduler_list:
            scheduler.step()

def aggregate_fedavg(w_locals):
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num

    (sample_num, averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params


"""model group 6 (classical VFL models): Host and Guest"""
# called by main_classic.py
class ClassificationModelHost(nn.Module):

    def __init__(self, encoder_local_bottom, encoder_local_top, encoder_cross, hidden_dim, num_classes,
                 use_encoder_cross=False, use_encoder_local=False, pool='mean', ratio=0.5, mlp_layer=1):
        super().__init__()
        self.ratio = ratio
        self.pool = pool
        self.use_encoder_cross = use_encoder_cross
        self.use_encoder_local = use_encoder_local
        self.backbone = nn.ModuleList()
        hidden_dim_ratio = 1
        if self.pool == 'concat':
            if use_encoder_local is True and use_encoder_cross is True:
                hidden_dim_ratio = 2
        hidden_dim = hidden_dim * hidden_dim_ratio
        
        # use_encoder_cross = True
        if self.use_encoder_cross:
            self.encoder_cross = copy.deepcopy(encoder_cross)
            self.backbone.append(self.encoder_cross)
            
        # use_encoder_local = False
        if self.use_encoder_local:
            self.encoder_local_bottom = copy.deepcopy(encoder_local_bottom)
            self.encoder_local_top = copy.deepcopy(encoder_local_top)
            self.backbone.append(self.encoder_local_bottom)
            self.backbone.append(self.encoder_local_top)

        #default num_cls_layer = 1
        if mlp_layer == 1:
            print('num_classes:{}'.format(num_classes))
            print('hidden_dim:{}'.format(hidden_dim))
            self.classifier_head = nn.Linear(hidden_dim, num_classes)
        elif mlp_layer == 2:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_dim[0], hidden_dim[1]),
                nn.ReLU(),
                nn.Linear(hidden_dim[1], num_classes)
            )
        elif mlp_layer == 3:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_dim[0], hidden_dim[1]),
                nn.ReLU(),
                nn.Linear(hidden_dim[1], hidden_dim[2]),
                nn.ReLU(),
                nn.Linear(hidden_dim[2], num_classes)
            )

    def forward(self, input_X):
        if self.use_encoder_cross:
            x_cross = self.encoder_cross(input_X).flatten(start_dim=1)
        if self.use_encoder_local:
            f = self.encoder_local_bottom(input_X)
            x_local = self.encoder_local_top(f).flatten(start_dim=1)
        if self.use_encoder_cross and self.use_encoder_local:
            if self.pool == 'mean':
                z = x_cross * self.ratio + x_local * (1 - self.ratio)
            elif self.pool == 'concat':
                z = torch.cat([x_cross, x_local], dim=1)
            else:
                z = x_cross * self.ratio + x_local * (1 - self.ratio)
        elif self.use_encoder_cross:
            z = x_cross
        elif self.use_encoder_local:
            z = x_local
        else:
            raise Exception
        return z

    def get_prediction(self, z_0, z_list):
        if z_list is not None:
            out = torch.cat([z_0] + z_list, dim=1)
        else:
            out = z_0
        x = self.classifier_head(out)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def load_encoder_cross(self, load_path, device):
        self.encoder_cross.load_state_dict(torch.load(load_path, map_location=device))

    def load_encoder_local_top(self, load_path, device):
        self.encoder_local_top.load_state_dict(torch.load(load_path, map_location=device))

    def load_encoder_local_bottom(self, load_path, device):
        self.encoder_local_bottom.load_state_dict(torch.load(load_path, map_location=device))

# called by main_cls.py
class ClassificationModelGuest(nn.Module):

    def __init__(self, encoder_local_bottom, encoder_local_top, encoder_cross, use_encoder_cross=False, use_encoder_local=False,
                 pool='mean', ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.pool = pool
        self.use_encoder_cross = use_encoder_cross
        self.use_encoder_local = use_encoder_local
        self.backbone = nn.ModuleList()
        print('Before append - backbone:{}'.format(self.backbone))

        if self.use_encoder_cross:
            self.encoder_cross = copy.deepcopy(encoder_cross)
            self.backbone.append(self.encoder_cross)

        if self.use_encoder_local:
            self.encoder_local_bottom = copy.deepcopy(encoder_local_bottom)
            self.encoder_local_top = copy.deepcopy(encoder_local_top)
            self.backbone.append(self.encoder_local_bottom)
            self.backbone.append(self.encoder_local_top)

    def forward(self, input_X):
        if self.use_encoder_cross:
            x_cross = self.encoder_cross(input_X).flatten(start_dim=1)
        if self.use_encoder_local:
            f = self.encoder_local_bottom(input_X)
            x_local = self.encoder_local_top(f).flatten(start_dim=1)
        if self.use_encoder_cross and self.use_encoder_local:
            if self.pool == 'mean':
                z = x_cross * self.ratio + x_local * (1 - self.ratio)
            elif self.pool == 'concat':
                z = torch.cat([x_cross, x_local], dim=1)
            else:
                z = x_cross * self.ratio + x_local * (1 - self.ratio)
        elif self.use_encoder_cross:
            z = x_cross
        elif self.use_encoder_local:
            z = x_local
        else:
            raise Exception
        return z

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def load_encoder_cross(self, load_path, device):
        self.encoder_cross.load_state_dict(torch.load(load_path, map_location=device))

    def load_encoder_local_top(self, load_path, device):
        self.encoder_local_top.load_state_dict(torch.load(load_path, map_location=device))

    def load_encoder_local_bottom(self, load_path, device):
        self.encoder_local_bottom.load_state_dict(torch.load(load_path, map_location=device))

    def save_models(self, target_dir, dataset, name_str):
        target_dir = os.path.join(target_dir, dataset)
        os.makedirs(target_dir, exist_ok=True)

        if self.use_encoder_cross:
            torch.save(self.encoder_cross.state_dict(),
                       os.path.join(target_dir, 'model_encoder_cross-{}.pth'.format(name_str)))
        if self.use_encoder_local:
            torch.save(self.encoder_local_bottom.state_dict(),
                       os.path.join(target_dir, 'model_encoder_local_bottom-{}.pth'.format(name_str)))
            torch.save(self.encoder_local_top.state_dict(),
                       os.path.join(target_dir, 'model_encoder_local_top-{}.pth'.format(name_str)))

