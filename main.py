import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

import torch.optim as optim
import torch.nn as nn
from SimpleCNN import SimpleCNN
from Train import train_model
import GlobalSetting


class carDataset(Dataset):
    def __init__(self, root, imgdir, labelfile, split, transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform

        self.image_label = pd.read_csv(f'{root}/{labelfile}')
        le = LabelEncoder()
        self.image_label["int_label"] = le.fit_transform(
            self.image_label["label"])

        self.imgspath = []
        for i in self.image_label['id']:
            i = "%06d" % i
            self.imgspath.append(f'{root}/{imgdir}/{i}.jpg')

        print('Total data in {} split: {}'.format(split, len(self.imgspath)))

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath = self.imgspath[index]
        image = Image.open(imgpath).convert('RGB')
        image = self.transform(image)
        label = torch.from_numpy(
                np.array(self.image_label.loc[index]['int_label']))

        # image = image.to(GlobalSetting.device)
        # label = label.to(GlobalSetting.device)
        return image, label

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgspath)


if __name__ == '__main__':
    # Convert a PIL image or numpy.ndarray to tensor.
    # (H*W*C) in range [0, 255] to a shape (C*H*W) in the range [0.0, 1.0].
    transform = transforms.Compose([
        transforms.Resize((500, 700)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=[0.4706, 0.4598, 0.4545],
        #                     std=[0.2628, 0.2616, 0.2663]),
    ])

    # Download train dataset
    trainSet = carDataset(root='./data', imgdir='training_data/training_data',
                          labelfile='training_labels.csv',
                          split='train', transform=transform)
    trainLoader = DataLoader(trainSet, batch_size=GlobalSetting.batch_size,
                             shuffle=True, num_workers=4)

    # net = SimpleCNN()
    net = GlobalSetting.Model
    net.to(GlobalSetting.device)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     patience=3,
                                                     verbose=1, min_lr=0.00001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train the network
    epochs = GlobalSetting.Epochs
    model = train_model(model=net,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        num_epochs=epochs,
                        doPlot=True,
                        trainSet=trainSet,
                        trainLoader=trainLoader)

    # Save the trained model
    torch.save(model.state_dict(), GlobalSetting.FinalPATH)
