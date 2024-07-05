import os
import pickle

from .utils import *

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
from .oxford_pets import OxfordPets

import torch
import codecs
import numpy as np
import sys

classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

class MNIST(DatasetBase):

    dataset_dir = "mnist"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir

        trainval_image_file = "train-images-idx3-ubyte"
        trainval_data = read_image_file(os.path.join(self.image_dir, trainval_image_file))  # Size([60000, 28, 28]) torch.uint8
        trainval_label_file = "train-labels-idx1-ubyte"
        trainval_targets = read_label_file(os.path.join(self.image_dir, trainval_label_file))  # Size([60000]) torch.int64
        # trainval_names = 

        test_image_file = "t10k-images-idx3-ubyte"
        test_data = read_image_file(os.path.join(self.image_dir, test_image_file))
        test_label_file = "t10k-labels-idx1-ubyte"
        test_targets = read_label_file(os.path.join(self.image_dir, test_label_file))

        trainval = []
        for idx in range(trainval_data.size(0)):
            item = Datum(impath=Image.fromarray(trainval_data[idx].numpy(), mode="L"), 
                         label=int(trainval_targets[idx]), classname=classes[trainval_targets[idx]])
            trainval.append(item)
        
        test = []
        for idx in range(test_data.size(0)):
            item = Datum(impath=Image.fromarray(test_data[idx].numpy(), mode="L"), 
                         label=int(test_targets[idx]), classname=classes[test_targets[idx]])
            test.append(item)

        train, val = OxfordPets.split_trainval(trainval)

        if num_shots >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

        subsample = subsample_classes
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        self.templates = [
            lambda c: f'a photo of the number: "{c}".',
        ]

        super().__init__(train_x=train, val=val, test=test)


def _flip_byte_order(t: torch.Tensor) -> torch.Tensor:
    return (
        t.contiguous().view(torch.uint8).view(*t.shape, t.element_size()).flip(-1).view(*t.shape[:-1], -1).view(t.dtype)
    )

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)

SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}

def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))

    # The MNIST format uses the big endian byte order, while `torch.frombuffer` uses whatever the system uses. In case
    # that is little endian and the dtype has more than one byte, we need to flip them.
    if sys.byteorder == "little" and parsed.element_size() > 1:
        parsed = _flip_byte_order(parsed)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 1:
        raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 3:
        raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
    return x