For training, we use the training set of ID dataset ImageNet-1K. For evaluation, we use the validation set of ImageNet-1K and four OOD datasets including iNaturalist, SUN, Places, and Texture. Due to the large size of these datasets, only a subset is provided here as an example. The directory structure for training is organized as follows:
|-- data
    |-- imagenet
        |-- classes.txt
        |-- images/
            |--train/ # 1,000 folders like n01484850, etc.
            |-- val/ # 1,000 folders like n01484850, etc.
    |-- iNaturalist
        |-- images/
    |-- SUN
    |-- Places
    |-- Texture
    ...