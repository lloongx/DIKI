import os
import pickle
import numpy as np

from .utils import *

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


class CIFAR100(DatasetBase):

    dataset_dir = "cifar100"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        file_path = os.path.join(root, self.dataset_dir, 'train')
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            trainval_data = entry["data"]
            if "labels" in entry:
                trainval_targets = entry["labels"]
            else:
                trainval_targets = entry["fine_labels"]
        trainval_data = trainval_data.reshape(-1, 3, 32, 32)
        trainval_data = trainval_data.transpose((0, 2, 3, 1))  # convert to HWC

        file_path = os.path.join(root, self.dataset_dir, 'test')
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            test_data = entry["data"]
            if "labels" in entry:
                test_targets = entry["labels"]
            else:
                test_targets = entry["fine_labels"]
        test_data = test_data.reshape(-1, 3, 32, 32)
        test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC

        path = os.path.join(self.dataset_dir, "meta")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            classes = data["fine_label_names"]
        classes = [s.replace("_", " ") for s in classes]

        trainval = []
        for idx in range(trainval_data.shape[0]):
            item = Datum(impath=Image.fromarray(trainval_data[idx]), 
                         label=int(trainval_targets[idx]), classname=classes[trainval_targets[idx]])
            trainval.append(item)
        
        test = []
        for idx in range(test_data.shape[0]):
            item = Datum(impath=Image.fromarray(test_data[idx]), 
                         label=int(test_targets[idx]), classname=classes[test_targets[idx]])
            test.append(item)
        
        train, val = OxfordPets.split_trainval(trainval)
        
        if num_shots >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

        subsample = subsample_classes
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        self.templates = [
            lambda c : f'a photo of a {c}.',
            lambda c : f'a blurry photo of a {c}.',
            lambda c : f'a black and white photo of a {c}.',
            lambda c : f'a low contrast photo of a {c}.',
            lambda c : f'a high contrast photo of a {c}.',
            lambda c : f'a bad photo of a {c}.',
            lambda c : f'a good photo of a {c}.',
            lambda c : f'a photo of a small {c}.',
            lambda c : f'a photo of a big {c}.',
            lambda c : f'a photo of the {c}.',
            lambda c : f'a blurry photo of the {c}.',
            lambda c : f'a black and white photo of the {c}.',
            lambda c : f'a low contrast photo of the {c}.',
            lambda c : f'a high contrast photo of the {c}.',
            lambda c : f'a bad photo of the {c}.',
            lambda c : f'a good photo of the {c}.',
            lambda c : f'a photo of the small {c}.',
            lambda c : f'a photo of the big {c}.',
        ]

        super().__init__(train_x=train, val=val, test=test)
