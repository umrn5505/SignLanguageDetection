import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataSet = pd.read_csv('archive/training_data.csv')
dataSet = np.array(dataSet)

X = np.array(dataSet[:,1:])
X = np.array(X) / X.max()
Y = np.array(dataSet[:,0])

unique_labels = np.unique(Y)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
Y = np.array([label_mapping[label] for label in Y])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.2, random_state=42)


class dataLoader(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float).to(device)
        self.Y = torch.tensor(Y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


trainingData = dataLoader(X_train, Y_train)
testingData = dataLoader(X_test, Y_test)
validationData= dataLoader(X_val, Y_val)

trainDataLoader = DataLoader(trainingData, batch_size=64, shuffle=True)
testDataLoader = DataLoader(testingData, batch_size=64, shuffle=True)
valDataLoader = DataLoader(validationData, batch_size=64, shuffle=True)


inputLayer = X.shape[1]
hiddenLayer1 = 128
hiddenLayer2 = 256
hiddenLayer3 = 128
hiddenLayer4 = 256
hiddenLayer5 = 128
outputLayer = len(np.unique(Y))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input = nn.Linear(inputLayer, hiddenLayer1)
        self.hidden1 = nn.Linear(hiddenLayer1, hiddenLayer2)
        self.hidden2 = nn.Linear(hiddenLayer2, hiddenLayer3)
        self.hidden3 = nn.Linear(hiddenLayer3, hiddenLayer4)
        self.hidden4 = nn.Linear(hiddenLayer4, hiddenLayer5)
        self.output = nn.Linear(hiddenLayer5, outputLayer)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.hidden1(x))
        x = self.dropout(self.relu(self.hidden2(x)))
        x = self.relu(self.hidden3(x))
        x = self.dropout(self.relu(self.hidden4(x)))
        x = self.output(x)
        return x


myModel = Model().to(device)


epochs = 200
criterion = nn.CrossEntropyLoss()
optimizer = Adam(myModel.parameters(), lr=0.001)

totalLossTrainPlot = []
totalLossValPlot = []
totalAccuracyTrainPlot = []
totalAccuracyValPlot = []
totalLossTestPlot = []
totalAccuracyTestPlot = []

for epoch in range(epochs):
    totalAccTrain = 0
    totalAccVal = 0
    totalLossTrain = 0
    totalLossVal = 0

    myModel.train()
    for data in trainDataLoader:
        inputs, labels = data

        predictions = myModel(inputs)
        loss = criterion(predictions, labels)
        totalLossTrain += loss.item()
        acc = (torch.argmax(predictions, dim=1) == labels).sum().item()
        totalAccTrain += acc
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    myModel.eval()
    with torch.no_grad():
        for data in valDataLoader:
            inputs, labels = data
            predictions = myModel(inputs)
            loss = criterion(predictions, labels)
            totalLossVal += loss.item()
            acc = (torch.argmax(predictions, dim=1) == labels).sum().item()
            totalAccVal += acc

    avgTrainLoss = totalLossTrain / len(trainDataLoader)
    avgValLoss = totalLossVal / len(valDataLoader)
    trainAccuracy = totalAccTrain / len(trainingData)
    valAccuracy = totalAccVal / len(validationData)

    totalLossTrainPlot.append(avgTrainLoss)
    totalLossValPlot.append(avgValLoss)
    totalAccuracyTrainPlot.append(trainAccuracy)
    totalAccuracyValPlot.append(valAccuracy)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avgTrainLoss:.4f}, Val Loss: {avgValLoss:.4f}, Train Acc: {trainAccuracy:.4f}, Val Acc: {valAccuracy:.4f}")

# Test phase
totalAccTest = 0
totalLossTest = 0
myModel.eval()
with torch.no_grad():
    for data in testDataLoader:
        inputs, labels = data
        predictions = myModel(inputs)
        loss = criterion(predictions, labels)
        totalLossTest += loss.item()
        acc = (torch.argmax(predictions, dim=1) == labels).sum().item()
        totalAccTest += acc

avgTestLoss = totalLossTest / len(testDataLoader)
testAccuracy = totalAccTest / len(testingData)

print(f"Test Loss: {avgTestLoss:.4f}, Test Accuracy: {testAccuracy:.4f}")

model_save = {
    'model_state_dict': myModel.state_dict(),
    'label_mapping': label_mapping,
    'input_size': inputLayer,
    'hidden_sizes': [hiddenLayer1, hiddenLayer2, hiddenLayer3, hiddenLayer4, hiddenLayer5],
    'output_size': outputLayer,
    'x_max': X.max()
}
torch.save(model_save, 'handwriting_model.pth')