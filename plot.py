import numpy as np
import matplotlib.pyplot as plt

def PlotCurve(epochs, trainLossValue, trainAccValue, doTrainloss=True, doTrain=True):
    if(doTrainloss):
        # Plot training loss curve
        epoch_list = np.arange(0, epochs)
        fig, ax = plt.subplots()
        plt.plot(epoch_list, trainLossValue, label='Train Loss')
        plt.xticks(np.arange(1, epochs+1, 1))
        plt.xlabel('Epoch')
        plt.title('Training Loss Curve')
        ax.legend(loc='best')
        plt.savefig('TrainingLossCurve.png')
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
        plt.savefig('TrainingCurve.png')
        plt.show()

    return