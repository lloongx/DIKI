import argparse
import os
import sys

from DIKI.datasets import get_dataset
from DIKI.utils import get_transform, set_random_seed
from DIKI.setup_cfg import setup_cfg, print_args
from DIKI.DIKI import DIKI



def run_exp(cfg):
    device = [int(s) for s in cfg.gpu_id.split(',')]

    cfg.use_validation = cfg.use_validation

    train_dataset, classes_names, templates = get_dataset(cfg, split='train', transforms=get_transform(cfg))
    val_dataset, _, _ = get_dataset(cfg, split='val', transforms=get_transform(cfg))
    eval_dataset, _, _ = get_dataset(cfg, split='test', transforms=get_transform(cfg))
    cfg.nb_task = len(eval_dataset)

    load_file = cfg.load_file if cfg.load_file else None
    trainer = DIKI(cfg, device, classes_names, templates, load_file=load_file)

    datasets = {'train': train_dataset, 'val': val_dataset, 'test': eval_dataset}
    trainer.train_and_eval(cfg, datasets)


def main(args):
    cfg = setup_cfg(args)
    cfg.command = ' '.join(sys.argv)

    shot = f'-FS' if cfg.num_shots > 0 else ''
    cfg.log_path = os.path.join('experiments', cfg.scenario, f'{cfg.dataset}'+shot)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    with open(os.path.join(cfg.log_path, 'config.yaml'), 'w') as f: 
        f.write(cfg.dump())
    
    print_args(args, cfg)

    set_random_seed(cfg.seed)
    run_exp(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs/MTIL.yaml", help="path to config")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)