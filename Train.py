import torch
from SimpleCNN import SimpleCNN
from plot import PlotCurve

import GlobalSetting

def train_model(model, criterion, optimizer, scheduler, num_epochs, doPlot, trainSet, trainLoader, path):
    trainLossValue = []
    trainAccValue = []
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        runningLoss = 0.0
        runningAcc = 0.0
        for i, data in enumerate(trainLoader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(GlobalSetting.device)
            labels = labels.to(GlobalSetting.device)

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
            runningAcc += (predicted == labels).sum().item()
        
        scheduler.step(runningLoss)

        # Print statistics
        runningAcc = runningAcc / len(trainSet)
        runningLoss = runningLoss / len(trainSet)
        print('[Epoch %d] Train loss: %.3f acc: %.3f' %(epoch + 1, runningLoss, runningAcc))
        trainAccValue.append(runningAcc)
        trainLossValue.append(runningLoss)

        torch.save(model.state_dict(), path)

    # Plot the curve of loss and accuracy
    if doPlot:
        PlotCurve(num_epochs, trainLossValue, trainAccValue, doVal=False)
    
    return model

def train_model_val(model, criterion, optimizer, scheduler, num_epochs, doPlot, trainSet, trainLoader, valSet, valLoader, path):
    trainLossValue = []
    trainAccValue = []
    valAccValue = []
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        runningLoss = 0.0
        runningAcc = 0.0
        valAcc = 0.0
        trainTotal = 0
        valTotal = 0
        model.train()
        for i, data in enumerate(trainLoader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(GlobalSetting.device)
            labels = labels.to(GlobalSetting.device)

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
            trainTotal += labels.size(0)
            runningAcc += (predicted == labels).sum().item()
        
        scheduler.step(runningLoss)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valLoader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(GlobalSetting.device)
                labels = labels.to(GlobalSetting.device)

                # Forward
                outputs = model(inputs)

                # Get prediction (values, indices)
                _, predicted = torch.max(outputs.data, 1)

                valTotal += labels.size(0)
                valAcc += (predicted == labels).sum().item()

        # Print statistics
        runningAcc = runningAcc / trainTotal
        runningLoss = runningLoss / trainTotal
        valAcc = valAcc / valTotal
        print('[Epoch %d] Train loss: %.3f acc: %.3f' %(epoch + 1, runningLoss, runningAcc))
        print('           Val acc: %.3f' %(valAcc))
        trainAccValue.append(runningAcc)
        trainLossValue.append(runningLoss)
        
        previousValue = valAccValue[-1]
        valAccValue.append(valAcc)

        if valAcc >= previousValue:        
            print("Store path")
            torch.save(model.state_dict(), path)

    # Plot the curve of loss and accuracy
    if doPlot:
        PlotCurve(num_epochs, trainLossValue, trainAccValue, valAccValue)
    
    return model