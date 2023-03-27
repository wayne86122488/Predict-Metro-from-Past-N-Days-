import numpy as np
import pandas as pd 
import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pandas_ods_reader import read_ods


all_data = []
for year in range(2018,2023,1):
      for month in range(1,10,1):
            path = './data/'+str(year)+'0'+str(month)+'_cht.ods'
            sheet_index = 1
            df = read_ods(path , sheet_index)
            data = df['台北車站'].to_numpy()
            all_data = np.concatenate((all_data,data))
      for month in range(10,13,1):
            path = './data/'+str(year)+str(month)+'_cht.ods'
            sheet_index = 1
            df = read_ods(path , sheet_index)
            data = df['台北車站'].to_numpy()
            all_data = np.concatenate((all_data,data))

class CustomData(Dataset):
      def __init__(self):
            self.datanumber = 1800
      
      def __len__(self):
            return self.datanumber

      def __getitem__(self, idx):
            X = all_data[idx:idx+7]
            y = all_data[idx+7]
            return X, y

metro_dataset = CustomData()
train_dataloader = DataLoader(metro_dataset, batch_size = 1000, shuffle=True)
test_dataloader = DataLoader(metro_dataset, batch_size = 1000, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
      def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                  nn.Linear(7, 7),
                  nn.Softmax(),
                  nn.Linear(7, 1),
                  nn.Sigmoid(),
                  nn.Linear(1, 1)
            )
      def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
      size = len(dataloader.dataset)
      model.train()
      for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error
            pred = model(X.to(torch.float32))
            loss = loss_fn(pred, y.to(torch.float32))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                  loss, current = loss.item(), (batch + 1) * len(X)
                  print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      model.eval()
      test_loss, correct = 0, 0
      with torch.no_grad():
            for X, y in dataloader:
                  X, y = X.to(device), y.to(device)
                  pred = model(X.to(torch.float32))
                  test_loss += loss_fn(pred, y.to(torch.float32)).item()
                  correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      test_loss /= num_batches
      correct /= size
      print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 1000
for t in range(epochs):
    print(f"Epoch {t}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

model.eval()
x, y = all_data[0:7], all_data[7]
x = torch.from_numpy(x)
with torch.no_grad():
    pred = model(x.to(torch.float32))
    predicted, actual = pred[0], y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')