import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np

import torch.optim as optim
import torch.nn as nn
from SimpleCNN import SimpleCNN
from Train import train_model

# Use GPU
if torch.cuda.is_available():  
  device = torch.device("cuda:8")  
else:  
  device = torch.device("cpu")

class carDataset(Dataset):
    def __init__(self, root, imgdir, labelfile, split, transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform

        self.image_label = pd.read_csv(f'{root}/{labelfile}')
        le = LabelEncoder()
        self.image_label["int_label"] = le.fit_transform(self.image_label["label"])

        self.imgspath = []
        for i in self.image_label['id']:
            i = "%06d" % i
            self.imgspath.append(f'{root}/{imgdir}/{i}.jpg')
        
        print('Total data in {} split: {}'.format(split, len(self.image_label)))

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        
        imgpath = self.imgspath[index]
        image = Image.open(imgpath).convert('RGB')
        image = self.transform(image)
        label = torch.from_numpy(np.array(self.image_label.loc[index]['int_label']))

        # image = image.to(device)
        # label = label.to(device)
        return image, label

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.image_label)

if __name__ == '__main__':
    
    # Convert a PIL image or numpy.ndarray to tensor.
    # (H*W*C) in range [0, 255] to a torch.FloatTensor of shape (C*H*W) in the range [0.0, 1.0].
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Download train dataset
    trainSet = carDataset(root='./data', imgdir='training_data/training_data', labelfile='training_labels.csv', split='train', transform=transform)
    trainLoader = DataLoader(trainSet, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    net = SimpleCNN()
    net.to(device)
    print(device)
    PATH = './net.pth'

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=1, min_lr=0.00001)

    # Train the network
    epochs = 10
    model = train_model(model=net,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        num_epochs=epochs,
                        doPlot=True,
                        trainSet=trainSet,
                        trainLoader=trainLoader)

    # Save the trained model
    torch.save(model.state_dict(), PATH)
