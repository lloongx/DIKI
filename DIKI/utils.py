import os
import numpy as np
import random
import torch


from clip import clip


def cosine_schedule_warmup(total_step, value, final_value=0, warmup_step=0, warmup_value=0):
    if warmup_step > 0:
        warmup_schedule = np.linspace(warmup_value, value, warmup_step+2)[1:-1]
    else:
        warmup_schedule = np.array([])
    steps = np.arange(total_step - warmup_step)
    schedule = final_value + 0.5 * (value-final_value) * (1+np.cos(np.pi * steps / len(steps)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_step
    return schedule

class build_cosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        init_lr = 0
        final_lr = lr * 1e-3
        self.lrs = cosine_schedule_warmup(total_step, lr, final_lr, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr = self.lrs[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"]= lr
        self.lr=lr


def get_transform(cfg):
    return clip._transform(cfg.input_size[0])


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_loss_3d(pred, target):
    pred = pred / pred.norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)
    loss = torch.sum(pred*target, dim=2)
    loss = 1 - torch.mean(loss)
    return loss

def cal_MTIL_metrics(acc_list):
    acc_list = np.array(acc_list)
    acc_list *= 100
    avg = acc_list.mean(axis=0)
    last = np.array(acc_list[-1, :])
    transfer = np.array([np.mean([acc_list[j, i] for j in range(i)]) for i in range(1, acc_list.shape[1])])
    g = lambda x: np.around(x.mean(), decimals=1) if len(x) > 0 else -1
    f = lambda x: [np.around(i, decimals=1) for i in x]
    return {"transfer": {"transfer": f(transfer)}, "avg": {"avg": f(avg)}, "last": {"last": f(last)}, 
            "results_mean": {"transfer": g(transfer), "avg": g(avg), "last": g(last)}}

