import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import torch
import random
import numpy as np

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.voc2007_distill
import datasets.coco2014_distill
import datasets.nuswide_distill
import datasets.voc2007_partial
import datasets.coco2014_partial
import datasets.nuswide_trainset_gt

import trainers.zsclip
import trainers.Caption_distill_double
import trainers.Caption_dual

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.use_tta:
        cfg.TEST.use_tta = args.use_tta

    if args.pre_epoch:
        cfg.TRAIN.pre_epoch = args.pre_epoch

    if args.global_size:
        cfg.TRAINER.global_size = args.global_size

    if args.local_size:
        cfg.TRAINER.local_size = args.local_size

    if args.use_chatglm:
        cfg.DATASET.use_chatglm = args.use_chatglm

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed > -1:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.train_data:
        cfg.DATASET.TRAIN_DATA = args.train_data

    if args.lr:
        cfg.TRAINER.SETLR = args.lr

    if args.lr_mul:
        cfg.TRAINER.SETLR_MUL = args.lr_mul

    if args.noise:
        cfg.TRAINER.NOSIE = args.noise

    if args.train_epoch:
        cfg.OPTIM.MAX_EPOCH = args.train_epoch

    if args.sizename:
        cfg.SIZENAME = args.sizename

    cfg.EVAL_ONLY = args.eval_only

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.Caption = CN()
    cfg.TRAINER.Caption.N_CTX = 16  # number of context vectors
    cfg.TRAINER.Caption.CSC = False  # class-specific context
    cfg.TRAINER.Caption.CTX_INIT = ""  # initialization words
    cfg.TRAINER.Caption.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.Caption.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.Caption.GL_merge_rate = 0.5
    # cfg.TRAINER.Caption.fewshot_TaI_merge_rate = 0.6
    # cfg.TRAINER.Caption.partial_TaI_merge_rate = 0.9

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SAMPLE = 0 # Sample some of all datas, 0 for no sampling i.e. using all
    cfg.DATASET.partial_prob = 0.5
    cfg.DATASET.use_chatglm = 0

    cfg.TRAIN.LOSSFUNC = ""  # or "focal"
    cfg.TRAIN.IF_LEARN_SCALE = False
    cfg.TRAIN.IF_LEARN_spatial_SCALE = False
    cfg.TRAIN.spatial_SCALE_text = 50
    cfg.TRAIN.spatial_SCALE_image = 50
    cfg.TRAIN.IF_ablation = False
    cfg.TRAIN.Caption_num = 0
    cfg.TRAIN.pre_epoch = -1
    
    cfg.TEST.EVALUATOR_ACT = "softmax"  # or "sigmoid"
    cfg.TEST.SAVE_PREDS = ""
    cfg.TEST.use_tta = False
    
    # several param for spacific transform setting
    cfg.INPUT.random_resized_crop_scale = (0.8, 1.0)
    cfg.INPUT.cutout_proportion = 0.4
    cfg.INPUT.TRANSFORMS_TEST = ("resize", "center_crop", "normalize")


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        print('merge_from_file {}'.format(args.dataset_config_file))
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model("../PVP_finetune/output/{model_path}", epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        if trainer.cfg.TRAIN.LOSSFUNC == 'dualcoop':
            trainer.train()
        else:

            config_data = args.config_file.rstrip('.yaml').split('/')[-1]

            data_ = 'xxx'
            if config_data == 'rn50_coco2014':
                data_ = 'coco'
            elif config_data == 'rn50_voc2007':
                data_ = 'voc'
            elif config_data == 'rn50_nuswide':
                data_ = 'nuswide'

            trainer.load_model(f'../pvp_pretrain/output/pretrain_{data_}/{args.global_size}x{args.local_size}*{args.w}w', image_pretrain = args.image_prompt_pretrain_load, 
            text_pretrain = args.text_prompt_pretrain_load, epoch=args.pre_epoch)
            trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--w", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--global-size", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--pre-epoch", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--local-size", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--use-tta", default=False, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--use-chatglm", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--image_prompt_pretrain_load", action="store_true", help="do not load pretrain image prompt"
    )
    parser.add_argument(
        "--text_prompt_pretrain_load", action="store_true", help="do not load pretrain text prompt"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument("--train_data", type=str, default="../data/subset-5k.txt", help="train data file path")
    parser.add_argument("--sizename", type=str, default="", help="split size name")
    parser.add_argument("--lr", type=float, default=1e-4, help="lr")
    parser.add_argument("--lr_mul", type=float, default=0.1, help="lr_mul")
    parser.add_argument("--noise", type=float, default=0.04, help="noise")
    parser.add_argument("--train_epoch", type=int, default=40, help="train_epoch")

    args = parser.parse_args()
    main(args)