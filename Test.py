import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np

from BestCNN import BestCNN

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

# Convert a PIL image or numpy.ndarray to tensor.
# (H*W*C) in range [0, 255] to a torch.FloatTensor of shape (C*H*W) in the range [0.0, 1.0].
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download test dataset
testSet = carDataset(root='./data', imgdir='testing_data/testing_data', labelfile='training_labels.csv', split='test', transform=transform)
testLoader = DataLoader(testSet, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

# Load existed model
net = BestCNN()
PATH = './net.pth'
net.load_state_dict(torch.load(PATH))

net.eval()


image_label = pd.read_csv(f'{'./data'}/{'training_labels.csv'}')
le = LabelEncoder()
image_label["int_label"] = le.fit_transform(image_label["label"])
int_label = image_label.groupby("int_label")

# Calculate top-3 error rate
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        labels = labels.view(-1,1) # Reshape labels from [n] to [n, 1] to compare [n, k]
        
        int_label.get_group(predicted)["label"]

        

        correct += (predicted == labels).sum().item()

