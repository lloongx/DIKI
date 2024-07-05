# Dataset instructions

DIKI is trained and evaluated on MTIL benchmark from [ZSCL](https://github.com/Thunderbeee/ZSCL). However, for the better consistency, we apply the dataset splits from [CoOp](https://github.com/KaiyangZhou/CoOp).

We suggest putting all datasets under the same folder (say $DATA) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like
```
$DATA/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
|-- ...
```

The instructions to prepare each dataset are detailed below.

### Caltech101
- Create a folder named `caltech-101/` under `$DATA`.
- Download `101_ObjectCategories.tar.gz` from https://data.caltech.edu/records/mzrjq-6wc02 and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`. 

The directory structure should look like
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### OxfordPets
- Create a folder named `oxford_pets/` under `$DATA`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing). 

The directory structure should look like
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### StanfordCars
- Create a folder named `stanford_cars/` under `$DATA`.
- ~~Download the train images http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.~~
- ~~Download the test images http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.~~
- ~~Download the train labels https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.~~
- ~~Download the test labels http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.~~
- The original download link has been broken, please refer to https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616 for solution.
- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).

The directory structure should look like
```
stanford_cars/
|–– cars_test\
|–– cars_test_annos_withlabels.mat
|–– cars_train\
|–– devkit\
|–– split_zhou_StanfordCars.json
```

### Flowers102
- Create a folder named `oxford_flowers/` under `$DATA`.
- Download the images and labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz and https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectively.
- Download `cat_to_name.json` from [here](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing). 
- Download `split_zhou_OxfordFlowers.json` from [here](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing).

The directory structure should look like
```
oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `$DATA`, resulting in a folder named `$DATA/food-101/`.
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like
```
food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### FGVCAircraft
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like
```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### SUN397
- Create a folder named  `sun397/` under `$DATA`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `$DATA/sun397/`.
- Download `split_zhou_SUN397.json` from this [link](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing).

The directory structure should look like
```
sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### DTD
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `$DATA`. This should lead to `$DATA/dtd/`.
- Download `split_zhou_DescribableTextures.json` from this [link](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

The directory structure should look like
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Create a folder named `eurosat/` under `$DATA`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `$DATA/eurosat/`.
- Download `split_zhou_EuroSAT.json` from [here](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing).

The directory structure should look like
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### MNIST
- Create a folder named `mnist/` under `$DATA`.
- Download four files from http://yann.lecun.com/exdb/mnist/ and put them into `$DATA/mnist/`.

The directory structure should look like
```
mnist/
|–– t10k-images-idx3-ubyte
|–– t10k-labels-idx1-ubyte
|–– train-images-idx3-ubyte
|–– train-labels-idx1-ubyte
```

### CIFAR100
- Create a folder named `cifar100/` under `$DATA`.
- Download the dataset from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz and extract it to `$DATA/cifar100/`.

The directory structure should look like
```
cifar100/
|–– meta
|–– test
|–– train
```