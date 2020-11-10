# Car Brand Classification
A ResNet model using PyTorch to classify car brand in Kaggle competition [CS_T0828_HW1](https://www.kaggle.com/c/cs-t0828-2020-hw1/leaderboard).

## Hardware
Ubuntu 18.04 LTS

Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

1x GeForce RTX 2080 Ti

## Set Up
### Install Dependency
All requirements is detailed in requirements.txt.

    $ pip install -r requirements.txt

### Create Directory
Create directory for store model path, result images, and test result.

    $ python setup.py

### Coding Style
Use PEP8 guidelines.

    $ pycodestyle *.py

## Dataset
The data directory is structured as:
```
└── data 
    ├── testing_data ─ 5,000 test images
    ├── training_data ─ 11,185 training images
    └── training_labels.csv ─ training labels
```

## Train
Train model.

    $ python main.py

Change the `modelname` in [GlobalSetting.py](./GlobalSetting.py) to change the name saving model weights, and result plot in training.

## Inference
Use pretrained weights to make predictions on images.

    $ python Test.py

Change the `modelname` in [GlobalSetting.py](./GlobalSetting.py) to get the pretrained weights you want and the path storing the test result.
