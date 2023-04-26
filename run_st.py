# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from semilearn.algorithms import get_algorithm, name2alg
from semilearn.imb_algorithms import get_imb_algorithm
from semilearn.algorithms.utils import str2bool
from semilearn.core.utils import get_net_builder, get_logger, get_port, send_model_cuda, count_parameters, over_write_args_from_file, TBLog
from transformers import RobertaTokenizer

def get_config():
    parser = argparse.ArgumentParser(description='Semi-Supervised Learning')
    # parser_from_model = argparse.ArgumentParser(description='Semi-Supervised Learning arguments from model')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('-sn', '--save_name', type=str, default=None)
    parser.add_argument('--resume', action='store_false')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_false', help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--num_train_iter', type=int, default=None,
                        help='total number of training iterations')
    parser.add_argument('--num_warmup_iter', type=int, default=None,
                        help='cosine linear warmup iterations')
    parser.add_argument('--num_eval_iter', type=int, default=None,
                        help='evaluation frequency')
    parser.add_argument('--num_log_iter', type=int, default=None,
                        help='logging frequencu')
    parser.add_argument('-nl', '--num_labels', type=int, default=None)
    parser.add_argument('-bsz', '--batch_size', type=int, default=None)
    parser.add_argument('--uratio', type=int, default=None,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ema_m', type=float, default=None, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--layer_decay', type=float, default=None, help='layer-wise learning rate decay, default to 1.0 which means no layer decay')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default=None)
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--use_pretrain', type=str2bool, default=False)
    parser.add_argument('--pretrain_path', type=str, default=False)

    '''
    Algorithms Configurations
    '''  

    ## core algorithm setting
    parser.add_argument('-alg', '--algorithm', type=str, default=None, help='ssl algorithm')
    parser.add_argument('--use_cat', type=str2bool, default=False, help='use cat operation in algorithms')
    parser.add_argument('--use_amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip_grad', type=float, default=0.0)

    ## imbalance algorithm setting
    parser.add_argument('-imb_alg', '--imb_algorithm', type=str, default=None, help='imbalance ssl algorithm')

    '''
    Data Configurations
    '''

    ## standard setting configurations
    parser.add_argument('--custom_unlabeled_data_file', type=str, default=None)
    parser.add_argument('--custom_dev_data_file', type=str, default=None)
    parser.add_argument('--custom_test_data_file', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('-ds', '--dataset', type=str, default=None)
    parser.add_argument('-nc', '--num_classes', type=int, default=None)
    parser.add_argument('--train_sampler', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)

    ## imbalanced setting arguments
    parser.add_argument('--lb_imb_ratio', type=int, default=1, help="imbalance ratio of labeled data, default to 1")
    parser.add_argument('--ulb_imb_ratio', type=list, default=None, help="imbalance ratio of unlabeled data, default to None")
    parser.add_argument('--ulb_num_labels', type=int, default=None, help="number of labels for unlabeled data, used for determining the maximum number of labels in imbalanced setting")

    ## cv dataset arguments
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--crop_ratio', type=float, default=None)

    ## nlp dataset arguments 
    parser.add_argument('--max_length', type=int, default=None)

    ## speech dataset algorithms
    parser.add_argument('--max_length_seconds', type=float, default=None)
    parser.add_argument('--sample_rate', type=int, default=None)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--c', type=str, default='')

    # Set default values: only load values from the config file that are not None in args
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    # Set default values: only load values that are not in args
    for argument in name2alg[args.algorithm].get_argument():
        if argument.name[2:] not in args.__dict__:
            setattr(args, argument.name[2:], argument.default)
            # parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)
    # args = parser.parse_args()
    # over_write_args_from_file(args, args.c)
    # for k, v in vars(args).items():
    #     if k not in vars(args):
    #         setattr(args, k, v)

    return args



def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''

    assert args.num_train_iter % args.epoch == 0, \
        f"# total training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    if args.gpu == 'None':
        args.gpu = None
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    if args.imb_algorithm is not None:
        model = get_imb_algorithm(args, _net_builder, tb_log, logger)
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')


    label_to_word = {
        "yahoo_answers": {"0": "culture", "1": "science", "2": "health", "3": "education", "4": "computer", "5": "sports", "6": "business", "7": "music", "8": "family", "9": "politics"},
        "amazon_review": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
        "yelp_review": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
        "ag_news": {"0": "world", "1": "sports", "2": "business", "3": "tech"},
        "aclImdb": {"0": "great", "1": "terrible"},
        "reverse_SST-2": {"0": "great", "1": "terrible"},
    }
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    kwargs = {'add_prefix_space': True} if isinstance(tokenizer, RobertaTokenizer) else {}
    label_words = label_to_word[args.dataset]
    mlm_logits_to_cls_logits_tensor = torch.zeros(len(label_words), dtype=torch.long, requires_grad=False)
    for i, (k, word) in enumerate(label_words.items()):
        ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
        assert len(ids) == 1
        mlm_logits_to_cls_logits_tensor[i] = ids[0]
    
    model.model.mlm_logits_to_cls_logits_tensor = mlm_logits_to_cls_logits_tensor
    model.ema_model.mlm_logits_to_cls_logits_tensor = mlm_logits_to_cls_logits_tensor

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model)
    logger.info(f"Arguments: {model.args}")

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))    
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    if hasattr(model, 'warmup'):
        logger.info(("Warmup stage"))
        model.warmup()

    # START TRAINING of FixMatch
    logger.info("Model training")
    model.train()

    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    if hasattr(model, 'finetune'):
        logger.info("Finetune stage")
        model.finetune()

    if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)