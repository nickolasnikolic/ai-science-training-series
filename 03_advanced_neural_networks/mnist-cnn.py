
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

transform_data = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=False, transform=transform_data)
test_data = datasets.MNIST(root='data', train=False, download=False, transform=transform_data)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_layer_0 = nn.Conv2d(1, 6, 3, 1, padding=3)
        self.convolutional_layer_1 = nn.Conv2d(6, 16, 3, 1)
        self.full_connected_layer_0 = nn.Linear(16*5*5, 120)
        self.full_connected_layer_1 = nn.Linear(120, 84)
        self.full_connected_layer_2 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.convolutional_layer_0(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.convolutional_layer_1(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 400)
        X = F.relu(self.full_connected_layer_0(X))
        X = F.relu(self.full_connected_layer_1(X))
        X = F.log_softmax(self.full_connected_layer_2(X), dim=1)
        return X


model = ConvolutionalNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(
    model.parameters(), 
    lr=0.01,
    lr_decay=0.01,
    weight_decay=0.01,
    initial_accumulator_value=0.5,
    eps=1e-10,
    maximize=True,
    differentiable=True
    )

epochs = 10
for e in range(epochs):
    #train the model
    for X_train, y_train in train_loader:
        output = model(X_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward() #needed to calculate the gradients
        optimizer.step() #needed to update the weights

    print(f'Epoch {e} Loss: {loss.item()}')

    # Test the model
    with torch.no_grad():
        correct = 0
        for X_test, y_test in test_loader:
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
            correct += (predicted == y_test).sum()
        print(f'Epoch {e} Testing')
    loss = criterion(output, y_test)

model.eval()
with torch.no_grad(): #chops to jose
    predicted = model(test_data[random.randint(0, len(test_data))][0].view(1, 1, 28, 28))
    print(predicted.argmax())

print('done')