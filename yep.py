# preprocessing libraries
import pandas as pd
import numpy as np

# models
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv('housing.csv')
data['total_bedrooms'].fillna(data['total_bedrooms'].median(), inplace = True)
data = pd.get_dummies(data, columns = ['ocean_proximity'], prefix = 'ocean')
# finding all the columns with categorical data
cat_features = [col for col in data.columns if col.startswith('ocean')]

# converting them into tensors
cat_values = data[cat_features].values
cat_values = torch.tensor(cat_values, dtype = torch.float32)
num_features = [col for col in data.columns if data[col].dtype == float and col not in ('median_house_value')]

# standardising them
data[num_features] = (data[num_features] - data[num_features].mean()) / data[num_features].std()

# converting them into tensors
num_values = data[num_features].values
num_values = torch.tensor(num_values, dtype = torch.float32)
house_values = torch.tensor(data['median_house_value'].values, dtype = torch.float32).reshape(-1, 1)
trainX, testX, trainY, testY = train_test_split(torch.cat((num_values, cat_values), dim = 1), house_values, random_state = 0)

class HouseValuePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HouseValuePredictor, self).__init__()
        
        # defining the number of layers and their size
        self.fc1 = nn.Linear(input_dim, 128).type(torch.float32)
        self.bn1 = nn.BatchNorm1d(128).type(torch.float32)
        
        self.fc2 = nn.Linear(128, 32).type(torch.float32)
        self.bn2 = nn.BatchNorm1d(32).type(torch.float32)
        
        self.fc3 = nn.Linear(32, 1).type(torch.float32)
        
        # defining dropouts
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # using ReLU as the activation function
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# passing the input dimensions into the model
input_dim = num_values.shape[1] + cat_values.shape[1]
model = HouseValuePredictor(input_dim)

# defining the loss funtion and optimizer
criterion = nn.L1Loss() # mean_absolute_error
optimizer = optim.Adam(model.parameters(), lr = 0.1)

epochs = 1000
losses = []

# training the model
for epoch in range(epochs):
    optimizer.zero_grad()
    
    outputs = model(trainX)
    loss = criterion(outputs, trainY)
    loss.backward()
    
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs} - Loss : {loss.item():.2f}')

    losses.append(loss.item())


model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('housing.pt') # Save

#torch.save(model.state_dict(),"housing.pt")