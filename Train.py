import torch
from BestCNN import BestCNN
from plot import PlotCurve

def train_model(model, criterion, optimizer, scheduler, num_epochs, doPlot, trainSet, trainLoader):
    trainLossValue = []
    trainAccValue = []
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        runningLoss = 0.0
        runningAcc = 0.0
        for i, data in enumerate(trainLoader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(torch.device("cuda:8"))
            labels = labels.to(torch.device("cuda:8"))

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)

            # Get prediction (values, indices)
            _, predicted = torch.max(outputs.data, 1)

            # Backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            labels = labels.view(-1,1) # Reshape labels from [n] to [n, 1] to compare [n, k]
            runningAcc += (predicted == labels).sum().item()
        
        scheduler.step(runningLoss)

        # Print statistics
        runningLoss = runningLoss / len(trainSet)
        runningAcc = runningAcc / len(trainSet)
        print('[Epoch %d] Train loss: %.3f acc: %.3f' %(epoch + 1, runningLoss, runningAcc))
        trainLossValue.append(runningLoss)
        trainAccValue.append(runningAcc)
        runningLoss = 0.0
        runningAcc = 0.0

    # Plot the curve of loss and accuracy
    if doPlot:
        PlotCurve(num_epochs, trainLossValue, trainAccValue)
    
    return model