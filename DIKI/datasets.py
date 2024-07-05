import os
import torch.nn as nn
import torch

# Names here are aligned to MTIL benchmark
from .MTIL_datasets.caltech101 import Caltech101
from .MTIL_datasets.cifar100 import CIFAR100
from .MTIL_datasets.dtd import DescribableTextures as DTD
from .MTIL_datasets.eurosat import EuroSAT
from .MTIL_datasets.fgvc_aircraft import FGVCAircraft as Aircraft
from .MTIL_datasets.food101 import Food101 as Food
from .MTIL_datasets.mnist import MNIST
from .MTIL_datasets.oxford_flowers import OxfordFlowers as Flowers
from .MTIL_datasets.oxford_pets import OxfordPets as OxfordPet
from .MTIL_datasets.stanford_cars import StanfordCars
from .MTIL_datasets.sun397 import SUN397
from .MTIL_datasets.ucf101 import UCF101
from .MTIL_datasets.utils import DatasetWrapper


def get_dataset(cfg, split, transforms=None):
    if split == 'val' and (not cfg.use_validation):
        return None, None, None

    is_train = (split == 'train')
    templates = None

    if cfg.dataset == "MTIL":
        '''
            Note that we split dataset to 'train', 'val' and 'test',
            which is different from original MTIL benchmark paper.
        '''
        if cfg.num_shots >= 1:
            base_sets = [Aircraft, Caltech101, CIFAR100, DTD, Flowers, Food, StanfordCars, SUN397]
        else:
            if cfg.MTIL_order_2:
                base_sets = [StanfordCars, Food, MNIST, OxfordPet, Flowers, SUN397, Aircraft, Caltech101, DTD, EuroSAT, CIFAR100]
            else:
                base_sets = [Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397]
        if cfg.train_one_dataset >= 0:
            base_sets = base_sets[cfg.train_one_dataset: cfg.train_one_dataset+1]
        dataset = []
        classes_names = []
        templates = []
        for base_set in base_sets:
            base = base_set(cfg.dataset_root, num_shots=cfg.num_shots, seed=cfg.seed)
            classes_names.append(base.classnames)
            templates.append(base.template)
            if split == 'train':
                dataset.append(DatasetWrapper(base.train_x, transform=transforms, is_train=is_train))
            elif split == 'val':
                dataset.append(DatasetWrapper(base.val, transform=transforms, is_train=is_train))
            elif split == 'test':
                dataset.append(DatasetWrapper(base.test, transform=transforms, is_train=is_train))
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset, classes_names, templates



def parse_sample(sample, is_train, task_id, cfg):
    return sample[0], sample[1], torch.IntTensor([task_id]).repeat(sample[0].size(0))