import torch
import torch.nn as nn

Test = True

# modelname = 'VGG19_pretrain_finetuneFC_epoch50'
# modelname = 'GoogleNet_pretrain_finetuneFC_epoch80'
# modelname = 'ResNet50_pretrain_finetuneall_size57_val_norm_batch8_epoch80'
print(modelname)

TrainingLossCurve = './resultImg/TrainingLossCurve_' + modelname + '.png'
TrainingCurve = './resultImg/TrainingCurve_' + modelname + '.png'
AllCurve = './resultImg/AllCurve_' + modelname + '.png'
TestResultPath = './testResult/' + modelname + 'Output.csv'
ValModelPath = './modelPath/val/' + modelname + '.pth'
FinalPATH = './modelPath/final_' + modelname + '.pth'
Checkpoints = './checkpoints/' + modelname + '/'
# Checkpoint for test
TestModelPath = './modelPath/val/' + modelname + '.pth'

Epochs = 80
batch_size = 8

Model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

# ResNet
for param in Model.parameters():
    param.requires_grad = True

# for param in Model.layer4.parameters():
#     param.requires_grad = True

num_fc_ftr = Model.fc.in_features
Model.fc = nn.Linear(num_fc_ftr, 196)

# # VGG
# for param in Model.parameters():
#         param.requires_grad = False

# for param in Model.classifier.parameters():
#     param.requires_grad = True

# num_fc_ftr = Model.classifier[6].in_features
# Model.classifier[6] = nn.Linear(num_fc_ftr, 196)

# # GoogleNet
# for param in Model.parameters():
#         param.requires_grad = False

# for param in Model.inception5b.parameters():
#     param.requires_grad = True

# num_fc_ftr = Model.fc.in_features
# Model.fc = nn.Linear(num_fc_ftr, 196)

# DenseNet
# Model = torch.hub.load('pytorch/vision:v0.6.0',
#                         'densenet121', pretrained=True)
# for param in Model.parameters():
#         param.requires_grad = False

# for param in Model.features.denseblock4.parameters():
#     param.requires_grad = True

# num_fc_ftr = Model.classifier.in_features
# Model.classifier = nn.Linear(num_fc_ftr, 196)

# Use GPU
if torch.cuda.is_available():
    device = torch.device("cuda:8")
else:
    device = torch.device("cpu")

print(device)
