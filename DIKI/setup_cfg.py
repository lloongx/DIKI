from yacs.config import CfgNode as CN


def reset_cfg(cfg, args):
    cfg.config_path = args.config_path
    cfg.gpu_id = args.gpu_id
    

def extend_cfg(cfg):
    """
    Add config variables.
    """
    cfg.dataset_root = ""
    cfg.model_backbone_name = ""
    cfg.input_size = (-1, -1)
    cfg.prompt_template = ""
    cfg.scenario = ""
    cfg.dataset = ""
    cfg.num_shots = -1
    cfg.seed = -1
    cfg.use_validation = False
    cfg.load_file = ""
    cfg.eval_only = False

    cfg.train_one_dataset = -1  # if >= 0, then only train corresponding dataset in MTIL
    cfg.zero_shot = False
    cfg.MTIL_order_2 = False

    cfg.DIKI = CN()
    cfg.DIKI.prompt_depth_vision = 1
    cfg.DIKI.prompt_depth_text = 1
    cfg.DIKI.n_ctx_vision = 12
    cfg.DIKI.n_ctx_text = 12
    cfg.DIKI.optim = CN()
    cfg.DIKI.optim.batch_size = 64
    cfg.DIKI.optim.name = "SGD"
    cfg.DIKI.optim.lr = 0.05
    cfg.DIKI.optim.max_epoch = 10
    cfg.DIKI.optim.weight_decay = 0
    cfg.DIKI.optim.lr_scheduler = "cosine"
    cfg.DIKI.optim.warmup_epoch = 0
    cfg.DIKI.batchwise_prompt = False


def setup_cfg(args):
    cfg = CN()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config_path)

    # From input arguments
    reset_cfg(cfg, args)

    # From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg


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