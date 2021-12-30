from torch._C import device
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


#!git clone https://github.com/beak2sm/ml.git
#!tar - zxvf ./ml/datasets/MNIST.tar.gz

path = './'
train_dataset = datasets.MNIST(path, train=True, download=True)
train_dataset = datasets.MNIST(path, train=False, download=True)

X_train, y_train = train_dataset.data / 255, train_dataset.targets
X_test, y_test = train_dataset.data / 255, train_dataset.targets

X_train, X_test = X_train.view(-1, 784), X_test.view(-1, 784)

train_dset = TensorDataset(X_train, y_train)
test_dset= TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dset,  batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset,  batch_size=32, shuffle=False)

class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(128, 10)
    def forward(self, X):
        out = self.hidden_layer1(X)
        out = self.hidden_layer2(out)
        out = self.output_layer(out)
        return out

device = 'cuda'

model = DNN(784).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0 #현재 에포크의 오차를 저장할 변수
    epoch_acc = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        hypothesis = model(X_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        y_predicted2 = torch.argmax(hypothesis, 1)
        acc = (y_predicted2==y_batch).float().mean()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    #오차를 다 합친 후에 나눠진 배치의 개수만큼 나눠 평균을 게산한다.
    return epoch_loss / len(loader), epoch_acc / len(loader)

n_epoch = 20
for epoch in range(0,n_epoch+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    if epoch % 10 == 0:
        print('epoch: {}, loss: {}, acc;{}'.format(epoch, loss, acc))  
