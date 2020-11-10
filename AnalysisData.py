import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob

# Plot the number of different brand in training data
image_label = pd.read_csv('./data/training_labels.csv')
le = LabelEncoder()
image_label["int_label"] = le.fit_transform(image_label["label"])

plt.figure()
plt.bar(image_label["int_label"].unique(), image_label["label"].value_counts(),
        width=0.3, align='center',
        color='lightblue', label="Number of training data")
plt.legend()
plt.xlabel('Number of training data')
plt.savefig('./trainingdata.png')


# Calculate mean and std of training data
class MyDataset(Dataset):
    def __init__(self):
        self.imgspath = []
        for imname in glob.glob('data/training_data/training_data/*.jpg'):
            # Run in all image in folder
            self.imgspath.append(imname)

        print('Total data: {}'.format(len(self.imgspath)))

    def __getitem__(self, index):
        imgpath = self.imgspath[index]
        image = Image.open(imgpath).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        image = transform(image)
        return image

    def __len__(self):
        return len(self.imgspath)


dataset = MyDataset()
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)

mean = 0.
std = 0.
nb_samples = 0.

for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print("Mean: {}".format(mean))
print("Std: {}".format(std))

# Calculate the average of width and height in training data
maxwidth = 0.0
maxheight = 0.0
minwidth = 1000.0
minheight = 1000.0
width = 0.0
height = 0.0
num = 0
for imname in glob.glob('data/training_data/training_data/*.jpg'):
    image = Image.open(imname).convert('RGB')
    imgwidth, imgheight = image.size
    width += imgwidth
    height += imgheight
    num += 1
    if maxwidth < imgwidth:
        maxwidth = imgwidth
    if maxheight < imgheight:
        maxheight = imgheight
    if minwidth > imgwidth:
        minwidth = imgwidth
    if minheight > imgheight:
        minheight = imgheight

print("Width: {}".format(width / num))
print("Height: {}".format(height / num))
print("Max Width: {}".format(maxwidth))
print("Max Height: {}".format(maxheight))
print("Min Width: {}".format(minwidth))
print("Min Height: {}".format(minheight))
