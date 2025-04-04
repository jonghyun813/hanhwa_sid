import logging.config
import os
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from configuration import config
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor

def main():
    args = config.base_parser()

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{args.dataset}/{args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{args.dataset}/{args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    #writer = SummaryWriter(f'tensorboard/{args.dataset}/{args.note}/seed_{args.rnd_seed}')

    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("Augmentation on GPU not available!")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    n_classes, image_path, label_path = get_statistics(dataset=args.dataset)
    print("args.use_human_training", args.use_human_training)

    logger.info(f"Select a CIL method ({args.mode})")
    if args.mode == 'ours' and args.weight_option == 'loss':
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, n_classes, device
    )
    
    # print()
    # print("###flops###")
    # method.get_flops_parameter()
    # print()

    eval_results = defaultdict(list)

    samples_cnt = 0
    task_id = 0

    # get datalist
    train_datalist, cls_dict, cls_addition = get_train_datalist(args.dataset, args.sigma, args.repeat, args.init_cls, args.rnd_seed)

    method.n_samples(len(train_datalist))

    # Reduce datalist in Debug mode
    if args.debug:
        random.shuffle(train_datalist)
        train_datalist = train_datalist[:5000]

    # _ = method.online_evaluate(samples_cnt, cls_dict, cls_addition, 0)

    #train_datalist = train_datalist[:5000] + train_datalist[10000:15000]
    print(f"total train stream: {len(train_datalist)}")

    for i, data in enumerate(train_datalist):

        # explicit task boundary for twf
        if samples_cnt % args.samples_per_task == 0 and (args.mode == "bic" or args.mode == "ewc++"):
            method.online_before_task(task_id)
            task_id += 1

        samples_cnt += 1
        # method.online_step(data, samples_cnt, args.n_worker)
        if samples_cnt == 1:
            import copy
            teacher_model = copy.deepcopy(method.model.model)
            teacher_learned_class = method.num_learned_class
        if samples_cnt % 100 == 0:
            print("UPDATING TEACHER MODEL")
            teacher_model = copy.deepcopy(method.model.model)
            teacher_learned_class = method.num_learned_class
        method.online_step(data, samples_cnt, args.n_worker, teacher_model, teacher_learned_class)

        '''
        if args.max_validation_interval is not None and args.min_validation_interval is not None:
            if samples_cnt % method.get_validation_interval() == 0:
                method.online_validate(samples_cnt, 512, args.n_worker)
        else:
            if samples_cnt % args.val_period == 0:
                method.online_validate(samples_cnt, 512, args.n_worker)
        '''

        '''
        ### for using validation set ###
        if samples_cnt % args.val_period == 0:
            method.online_validate(samples_cnt, 512, args.n_worker)
        ''' 
        if samples_cnt % args.eval_period == 0:
            eval_dict = method.online_evaluate(samples_cnt, cls_dict, cls_addition, data["time"])
            eval_results["test_acc"].append(eval_dict['avg_mAP50'])
            eval_results["classwise_acc"].append(eval_dict['classwise_mAP50'])
            eval_results["data_cnt"].append(samples_cnt)
            # if method.f_calculated:
            #     eval_results["forgetting_acc"].append(eval_dict['cls_acc'])
    if eval_results["data_cnt"][-1] != samples_cnt:
        eval_dict = method.online_evaluate(samples_cnt, cls_dict, cls_addition, data["time"])

    A_last = eval_dict['avg_mAP50']

    if args.mode == 'gdumb':
        eval_results = method.evaluate_all(args.memory_epoch, cls_dict, cls_addition)

    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval.npy', eval_results['test_acc'])
    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["avg_mAP50"])
    # A_online = np.mean(eval_results["online_acc"])

    # cls_acc = np.array(eval_results["forgetting_acc"])
    # acc_diff = []
    # for j in range(n_classes):
    #     if np.max(cls_acc[:-1, j]) > 0:
    #         acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
    # F_last = np.mean(acc_diff)
    # IF_avg = np.mean(method.forgetting[1:])
    # KG_avg = np.mean(method.knowledge_gain[1:])
    # Total_flops = method.get_total_flops()

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc} |  A_last {A_last} ") #| Total_flops {Total_flops}")

if __name__ == "__main__":
    main()