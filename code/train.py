import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import trainers.mnpt
import datasets.imagenet

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

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.topk:
        cfg.topk = args.topk

    cfg.miu_value = args.miu_value

    cfg.annealed_temperature = args.annealed_temperature

    if args.min_temperature:
        cfg.min_temperature = args.min_temperature
        
    if args.annealed_rate:
        cfg.annealed_rate = args.annealed_rate
    
    if args.init_temperature:
        cfg.init_temperature = args.init_temperature


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN
    
    cfg.TRAINER.MNPT = CN()
    cfg.TRAINER.MNPT.N_CTX = 16
    cfg.TRAINER.MNPT.CSC = False
    cfg.TRAINER.MNPT.PREC = "fp16"
    cfg.TRAINER.MNPT.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.MNPT.N_CTX_NEG = 16
    cfg.TRAINER.MNPT.N_CTX_NP = 100
    cfg.TRAINER.MNPT.POSITIVE_WEIGHTS = ""
    cfg.TRAINER.MNPT.N_CTX_FALLBACK = 16
    
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)

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

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)        
        return

    if not args.no_train:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str,
                        default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
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
    parser.add_argument("--trainer", type=str, default="",
                        help="name of trainer")
    parser.add_argument("--backbone", type=str, default="",
                        help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true",
                        help="evaluation only")
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
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--topk', type=int, default=200,
                        help='topk for extracted OOD regions')

    parser.add_argument('--miu_value', type=float, default=0.01,
                        help='weight for diversity loss')
    parser.add_argument('--annealed_temperature', action="store_true", 
                        help="annealed temperature if true, constant if false")
    parser.add_argument('--min_temperature', type=float, default=1, 
                        help="min temperature for annealed temperature")
    parser.add_argument('--annealed_rate', type=float, default=0.2, 
                        help="annealed rate for annealed temperature")
    parser.add_argument('--init_temperature', type=float, default=10, 
                        help="initial temperature for annealed temperature")
    args = parser.parse_args()
    main(args)
