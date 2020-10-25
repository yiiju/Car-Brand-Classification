import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob

from SimpleCNN import SimpleCNN

# Use GPU
if torch.cuda.is_available():  
  device = torch.device("cuda:8")  
else:  
  device = torch.device("cpu")

class carDataset(Dataset):
    def __init__(self, root, imgdir, split, transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform

        self.imgspath = []
        self.img_id = []
        for imname in glob.glob(f'{root}/{imgdir}' + '/*.jpg'):
            # Run in all image in folder
            self.imgspath.append(imname)
            self.img_id.append(imname.split('/')[-1].split('.')[0])
        
        print('Total data in {} split: {}'.format(split, len(self.img_id)))

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        
        imgpath = self.imgspath[index]
        image = Image.open(imgpath).convert('RGB')
        image = self.transform(image)
        imageid = self.img_id[index]

        return image, imageid

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.img_id)

# Convert a PIL image or numpy.ndarray to tensor.
# (H*W*C) in range [0, 255] to a torch.FloatTensor of shape (C*H*W) in the range [0.0, 1.0].
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download test dataset
testSet = carDataset(root='./data', imgdir='testing_data/testing_data', split='test', transform=transform)
testLoader = DataLoader(testSet, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

# Load existed model
net = SimpleCNN()
PATH = './net.pth'
net.load_state_dict(torch.load(PATH))
net.to(device)

net.eval()


image_label = pd.read_csv('./data/training_labels.csv')
le = LabelEncoder()
image_label["int_label"] = le.fit_transform(image_label["label"])

idary = []
labelary = []
# Calculate top-3 error rate
with torch.no_grad():
    for data in testLoader:
        images, imageid = data
        images = images.to(torch.device("cuda:8"))

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        
        idary = np.append(idary, imageid)    
        labelary = np.append(labelary, le.inverse_transform(predicted.cpu()))     

df = pd.DataFrame({'id': idary,
                   'label': labelary})

df.to_csv('./testOutput.csv', index=False)