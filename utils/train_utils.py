from torch import optim
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import SGD
import numpy as np
import copy
from utils.data_loader import get_train_datalist, ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, get_test_datalist
import torch.nn.functional as F
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor
from utils.my_augment import Kornia_Randaugment
from torchvision import transforms
from tqdm import tqdm

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i
            
class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def kl_loss(logits_stu, logits_tea, temperature=4.0):
    """
    Args:
        logits_stu: student logits
        logits_tea: teacher logits
        temperature: temperature
    Returns:
        distillation loss
    """
    pred_teacher = F.softmax(logits_tea / temperature, dim=1)
    log_pred_student = F.log_softmax(logits_stu / temperature, dim=1)
    loss_kd = F.kl_div(
        log_pred_student,
        pred_teacher,
        reduction='none'
    ).sum(1).mean(0) * (temperature ** 2)
    return loss_kd

def select_optimizer(name, model, lr=0.01, momentum=0.9, decay=1e-5,):
    # if hasattr(model, 'fc'):
    #     fc_name = 'fc'
    # elif hasattr(model, 'head'):
    #     fc_name = 'head'
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                g[2].append(param)
            elif isinstance(module, bn):  # weight (no decay)
                g[1].append(param)
            else:  # weight (with decay)
                g[0].append(param)

    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
    name = {x.lower(): x for x in optimizers}.get(name.lower())
    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(
            f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
            "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
        )

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

    return optimizer
    # if 'freeze_fc' not in opt_name:
    #     opt.add_param_group({'params': getattr(model, fc_name).parameters()})
    # return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler


def get_data_loader(opt_dict, dataset, pre_train=False):
    if pre_train:
        batch_size = 128
    else:
        batch_size = opt_dict['batchsize']

    # pre_dataset을 위한 dataset 불러오고 dataloader 생성
    train_transform, test_transform = get_transform(dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

    test_datalist = get_test_datalist(dataset)
    train_datalist, cls_dict, cls_addition = get_train_datalist(dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

    # for debugging!
    # train_datalist = train_datalist[:2000]

    exp_train_df = pd.DataFrame(train_datalist)
    exp_test_df = pd.DataFrame(test_datalist)

    train_dataset = ImageDataset(
        exp_train_df,
        dataset=dataset,
        transform=train_transform,
        preload = True,
        use_kornia=True,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=opt_dict["n_worker"],
    )

    test_dataset = ImageDataset(
        exp_test_df,
        dataset=dataset,
        transform=test_transform,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,#opt_dict["batchsize"],
        num_workers=opt_dict["n_worker"],
    )

    return train_loader, test_loader

def select_model(model_name, dataset, num_classes=None, opt_dict=None, G=False, F=False, ver2=False, channel_constant=1, kwinner = False, vit_pretrain=False):
    if 'vit' in model_name:
        print("vit_pretrain", vit_pretrain)
        return timm.create_model(model_name, pretrained=vit_pretrain) # vit_large_patch16_224, vit_base_patch16_224
    
    model_imagenet = False
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

    if "mnist" in dataset:
        model_class = getattr(mnist, "MLP")
    elif "cifar" in dataset or "5_dataset" in dataset:
        model_class = getattr(cifar, "ResNet")
        if G:
            model_class = getattr(cifar, "ResNet_G")
            opt["ver2"] = ver2
        elif F:
            model_class = getattr(cifar, "ResNet_F")
            opt["ver2"] = ver2
    elif "imagenet" in dataset or "clear" in dataset:
        #model_class = getattr(imagenet, "ResNet")
        model_imagenet=True
        model_class = getattr(cifar, "ResNet")
        if G:
            model_class = getattr(cifar, "ResNet_G")
            opt["ver2"] = ver2
        elif F:
            model_class = getattr(cifar, "ResNet_F")
            opt["ver2"] = ver2
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet, 5_dataset, clear)"
        )
    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet20":
        opt["depth"] = 20
    elif model_name == "resnet44":
        opt["depth"] = 44
    elif model_name == "resnet56":
        opt["depth"] = 56
    elif model_name == "resnet110":
        opt["depth"] = 110
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt, model_imagenet, channel_constant, kwinner)

    # TODO initial check
    initial = False

    # pre_dataset 설정
    pre_dataset = None
    pre_dataset_num_class = 0
    path_load_cp = None
    if dataset == "cifar10":
        pre_dataset = "cifar100"
        path_load_cp = "res18_cifar100_pretrained_model.pth" #"checkpoint/rs18_cifar100_new.pth"
        pre_dataset_num_class = 100
    elif dataset == "cifar100":
        pre_dataset = "tiny_imagenet"
        pre_dataset_num_class = 1000
    else:
        pre_dataset = "anything"

    assert pre_dataset is not None # none이면 설정이 덜 된것
    if opt_dict is not None:

        if initial:
            device = opt_dict['device']
            '''
            # pre_dataset을 위한 dataset 불러오고 dataloader 생성
            train_transform, test_transform = get_transform(pre_dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

            test_datalist = get_test_datalist(pre_dataset)
            train_datalist, cls_dict, cls_addition = get_train_datalist(pre_dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

            # for debugging!
            # train_datalist = train_datalist[:2000]

            exp_train_df = pd.DataFrame(train_datalist)
            exp_test_df = pd.DataFrame(test_datalist)

            train_dataset = ImageDataset(
                exp_train_df,
                dataset=pre_dataset,
                transform=train_transform,
                preload = True,
                use_kornia=True,
                #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
                data_dir=opt_dict["data_dir"]
            )
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=128,
                num_workers=opt_dict["n_worker"],
            )
            test_dataset = ImageDataset(
                exp_test_df,
                dataset=pre_dataset,
                transform=test_transform,
                #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
                data_dir=opt_dict["data_dir"]
            )
            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=128,#opt_dict["batchsize"],
                num_workers=opt_dict["n_worker"],
            )
            '''
            train_loader, test_loader = get_data_loader(opt_dict, pre_dataset, pre_train=True)

            model.fc = torch.nn.Linear(model.fc.in_features, pre_dataset_num_class).to(device)
            model.to(device)
            model.train()
            #opt = SGD(model.parameters(), lr=opt_dict["lr"], weight_decay=opt_dict["optim_wd"], momentum=opt_dict["optim_mom"])
            opt = optim.Adam(model.parameters(), lr=opt_dict["lr"])
            criterion = F.cross_entropy

            for epoch in range(opt_dict["pre_epoch"]):
            #for epoch in range(10):
                correct = 0
                num_data = 0
                total_loss = 0.0
                iteration = 0

                for i, data in tqdm(enumerate(train_loader), desc='Pre-training epoch {}'.format(epoch), leave=False, total=len(train_loader)):
                    model.train()
                    x = data["image"]
                    y = data["label"]
                    x = x.to(device)
                    y = y.to(device)
                    logit = model(x)
                    loss = criterion(logit, y)
                    loss.backward()
                    opt.step()

                    _, preds = logit.topk(1, 1, True, True)
                    correct += torch.sum(preds == y.unsqueeze(1)).item()
                    num_data += y.size(0)
                    total_loss += loss.item()
                    iteration+=1

                print(f"[TRAIN] epoch{epoch} loss",  total_loss / iteration, "accuracy", correct / num_data)

                if epoch % 10 == 0:
                    model.eval()
                    for i, data in enumerate(test_loader):
                        x = data["image"]
                        y = data["label"]
                        x = x.to(device)
                        y = y.to(device)
                        logit = model(x)
                        loss = criterion(logit, y)

                        _, preds = logit.topk(1, 1, True, True)
                        correct += torch.sum(preds == y.unsqueeze(1)).item()
                        num_data += y.size(0)
                        total_loss += loss.item()
                        iteration+=1

                    print("[TEST] epoch{epoch} loss",  total_loss / iteration, "accuracy", correct / num_data)

            torch.save(model.state_dict(), "res18_cifar100_pretrained_model.pth")        
        else:
            model = load_initial_checkpoint(pre_dataset, model, opt_dict["device"], load_cp_path = path_load_cp)
    return model

##### for ASER #####
def compute_knn_sv(model, eval_x, eval_y, cand_x, cand_y, k, device="cpu"):
    """
        Compute KNN SV of candidate data w.r.t. evaluation data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                eval_y (tensor): evaluation label tensor.
                cand_x (tensor): candidate data tensor.
                cand_y (tensor): candidate label tensor.
                k (int): number of nearest neighbours.
                device (str): device for tensor allocation.
            Returns
                sv_matrix (tensor): KNN Shapley value matrix of candidate data w.r.t. evaluation data.
    """
    # Compute KNN SV score for candidate samples w.r.t. evaluation samples
    n_eval = eval_x.size(0)
    n_cand = cand_x.size(0)
    # Initialize SV matrix to matrix of -1
    sv_matrix = torch.zeros((n_eval, n_cand), device=device)
    # Get deep features
    eval_df, cand_df = deep_features(model, eval_x, n_eval, cand_x, n_cand)
    # Sort indices based on distance in deep feature space
    sorted_ind_mat = sorted_cand_ind(eval_df, cand_df, n_eval, n_cand)

    # Evaluation set labels
    el = eval_y
    el_vec = el.repeat([n_cand, 1]).T
    # Sorted candidate set labels
    cl = cand_y[sorted_ind_mat]

    # Indicator function matrix
    indicator = (el_vec == cl).float()
    indicator_next = torch.zeros_like(indicator, device=device)
    indicator_next[:, 0:n_cand - 1] = indicator[:, 1:]
    indicator_diff = indicator - indicator_next

    cand_ind = torch.arange(n_cand, dtype=torch.float, device=device) + 1
    denom_factor = cand_ind.clone()
    denom_factor[:n_cand - 1] = denom_factor[:n_cand - 1] * k
    numer_factor = cand_ind.clone()
    numer_factor[k:n_cand - 1] = k
    numer_factor[n_cand - 1] = 1
    factor = numer_factor / denom_factor

    indicator_factor = indicator_diff * factor
    indicator_factor_cumsum = indicator_factor.flip(1).cumsum(1).flip(1)

    # Row indices
    row_ind = torch.arange(n_eval, device=device)
    row_mat = torch.repeat_interleave(row_ind, n_cand).reshape([n_eval, n_cand])

    # Compute SV recursively
    sv_matrix[row_mat, sorted_ind_mat] = indicator_factor_cumsum

    return sv_matrix


def deep_features(model, eval_x, n_eval, cand_x, n_cand):
    """
        Compute deep features of evaluation and candidate data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                n_eval (int): number of evaluation data.
                cand_x (tensor): candidate data tensor.
                n_cand (int): number of candidate data.
            Returns
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
    """
    # Get deep features
    if cand_x is None:
        num = n_eval
        total_x = eval_x
    else:
        num = n_eval + n_cand
        total_x = torch.cat((eval_x, cand_x), 0)

    # compute deep features with mini-batches
    total_x = maybe_cuda(total_x)
    deep_features_ = mini_batch_deep_features(model, total_x, num)

    eval_df = deep_features_[0:n_eval]
    cand_df = deep_features_[n_eval:]
    return eval_df, cand_df


def sorted_cand_ind(eval_df, cand_df, n_eval, n_cand):
    """
        Sort indices of candidate data according to
            their Euclidean distance to each evaluation data in deep feature space.
            Args:
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
                n_eval (int): number of evaluation data.
                n_cand (int): number of candidate data.
            Returns
                sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
    """
    # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
    # Preprocess feature vectors to facilitate vector-wise distance computation
    eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
    cand_df_tile = cand_df.repeat([n_eval, 1])
    # Compute distance between evaluation and candidate feature vectors
    distance_vector = euclidean_distance(eval_df_repeat, cand_df_tile)
    # Turn distance vector into distance matrix
    distance_matrix = distance_vector.reshape((n_eval, n_cand))
    # Sort candidate set indices based on distance
    sorted_cand_ind_ = distance_matrix.argsort(1)
    return sorted_cand_ind_


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction='mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean(0)

        return loss.mean() if self.reduction == 'mean' else loss.sum()

def re_init_weights(param_data, shape, device, reinint_method='xavier'):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        renint_usnig_method(param_data, mask, reinint_method)
        mask = torch.squeeze(mask, 1)
    else:
        renint_usnig_method(param_data, mask, reinint_method)
    return mask

def renint_usnig_method(param_data, mask, method='xavier'):
    if method == 'kaiming':
        nn.init.kaiming_uniform_(mask)
    elif method == 'normal':
        std, mean = torch.std_mean(param_data)
        nn.init.normal_(mask, mean, std)
    elif method == 'xavier':
        nn.init.xavier_uniform_(mask)

def create_dense_mask_0(net, device, value, fc=False):
    for name, param in net.named_parameters():
        if fc:
            if "fc" in name:
                param.data[param.data == param.data] = value
        else:
            param.data[param.data == param.data] = value
    net.to(device)
    return net

def test_sparsity(model, column=True, channel=True, filter=True, kernel=False):

    # --------------------- total sparsity --------------------
    total_zeros = 0
    total_nonzeros = 0
    layer_cont = 1

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):# and "shortcut" not in name):
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            print("(empty/total) weights of {}({}) is: ({}/{}). irregular sparsity is: {:.4f}".format(
                name, layer_cont, zeros, zeros+non_zeros, zeros / (zeros+non_zeros)))

        layer_cont += 1

    comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros)
    total_sparsity = total_zeros / (total_zeros + total_nonzeros)

    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / total_nonzeros))
    print("===========================================================================\n\n")

    # --------------------- column sparsity --------------------
    if(column):

        total_column = 0
        total_empty_column = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                weight2d = weight.reshape(weight.shape[0], -1)
                column_num = weight2d.shape[1]

                empty_column = np.sum(np.sum(np.absolute(weight2d.cpu().detach().numpy()), axis=0) == 0)
                print("(empty/total) column of {}({}) is: ({}/{}). column sparsity is: {:.4f}".format(
                    name, layer_cont, empty_column, weight.size()[1] * weight.size()[2] * weight.size()[3],
                                        empty_column / column_num))

                total_column += column_num
                total_empty_column += empty_column
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of column: {}, empty-column: {}, column sparsity is: {:.4f}".format(
            total_column, total_empty_column, total_empty_column / total_column))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")


    # --------------------- channel sparsity --------------------
    if (channel):

        total_channels = 0
        total_empty_channels = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                empty_channels = 0
                channel_num = weight.size()[1]

                for i in range(channel_num):
                    if np.sum(np.absolute(weight[:, i, :, :].cpu().detach().numpy())) == 0:
                        empty_channels += 1
                print("(empty/total) channel of {}({}) is: ({}/{}) ({}). channel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_channels, weight.size()[1], weight.size()[1]-empty_channels, empty_channels / channel_num))

                total_channels += channel_num
                total_empty_channels += empty_channels
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of channels: {}, empty-channels: {}, channel sparsity is: {:.4f}".format(
            total_channels, total_empty_channels, total_empty_channels / total_channels))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")


    # --------------------- filter sparsity --------------------
    if(filter):

        total_filters = 0
        total_empty_filters = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                empty_filters = 0
                filter_num = weight.size()[0]

                for i in range(filter_num):
                    if np.sum(np.absolute(weight[i, :, :, :].cpu().detach().numpy())) == 0:
                        empty_filters += 1
                print("(empty/total) filter of {}({}) is: ({}/{}) ({}). filter sparsity is: {:.4f} ({:.4f})".format(
                    name, layer_cont, empty_filters, weight.size()[0], weight.size()[0]-empty_filters, empty_filters / filter_num, 1-(empty_filters / filter_num)))

                total_filters += filter_num
                total_empty_filters += empty_filters
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of filters: {}, empty-filters: {}, filter sparsity is: {:.4f}".format(
            total_filters, total_empty_filters, total_empty_filters / total_filters))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")

    # --------------------- kernel sparsity --------------------
    if(kernel):

        total_kernels = 0
        total_empty_kernels = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                shape = weight.shape
                npWeight = weight.cpu().detach().numpy()
                weight3d = npWeight.reshape(shape[0], shape[1], -1)

                empty_kernels = 0
                kernel_num = weight.size()[0] * weight.size()[1]

                for i in range(weight.size()[0]):
                    for j in range(weight.size()[1]):
                        if np.sum(np.absolute(weight3d[i, j, :])) == 0:
                            empty_kernels += 1
                print("(empty/total) kernel of {}({}) is: ({}/{}) ({}). kernel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_kernels, kernel_num, kernel_num-empty_kernels, empty_kernels / kernel_num))

                total_kernels += kernel_num
                total_empty_kernels += empty_kernels
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of kernels: {}, empty-kernels: {}, kernel sparsity is: {:.4f}".format(
            total_kernels, total_empty_kernels, total_empty_kernels / total_kernels))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")
    return comp_ratio, total_sparsity

def copy_paste_fc(new_fc, prev_fc):
    prev_weight = copy.deepcopy(prev_fc.weight.data)
    prev_bias = copy.deepcopy(prev_fc.bias.data)
    with torch.no_grad():
        if new_fc.out_features > 1:
            new_fc.weight[:new_fc.out_features - 1] = prev_weight
            new_fc.bias[:new_fc.out_features - 1] = prev_bias
    return new_fc