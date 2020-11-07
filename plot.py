import numpy as np
import matplotlib.pyplot as plt

import GlobalSetting

def PlotCurve(epochs, trainLossValue=0, trainAccValue=0, valAccValue=0, doTrainloss=True, doTrain=True, doVal=True):
    if(doTrainloss):
        # Plot training loss curve
        epoch_list = np.arange(0, epochs)
        fig, ax = plt.subplots()
        plt.plot(epoch_list, trainLossValue, label='Train Loss')
        plt.xticks(np.arange(1, epochs+1, 1))
        plt.xlabel('Epoch')
        plt.title('Training Loss Curve')
        ax.legend(loc='best')
        plt.savefig(GlobalSetting.TrainingLossCurve)
        plt.show()

    if(doTrain):
        # Plot training curve
        epoch_list = np.arange(0, epochs)
        fig, ax = plt.subplots()
        plt.plot(epoch_list, trainAccValue, label='Train Accuracy')
        plt.plot(epoch_list, trainLossValue, label='Train Loss')
        plt.xticks(np.arange(1, epochs+1, 1))
        plt.xlabel('Epoch')
        plt.title('Training Curve')
        ax.legend(loc='best')
        plt.savefig(GlobalSetting.TrainingCurve)
        plt.show()

    if(doVal):
        # Plot training curve
        epoch_list = np.arange(0, epochs)
        fig, ax = plt.subplots()
        plt.plot(epoch_list, trainAccValue, label='Train Accuracy')
        plt.plot(epoch_list, valAccValue, label='Validation Accuracy')
        plt.xticks(np.arange(1, epochs+1, 1))
        plt.xlabel('Epoch')
        plt.title('Training and Validation Curve')
        ax.legend(loc='best')
        plt.savefig(GlobalSetting.AllCurve)
        plt.show()

    return