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

#정규화과정이다. 흑백데이터는 픽셀값으로 0에서 255의 값을 갖는데 225로
#나눠 0에서 1사이 값을 갖도록 정규화 한 것이다.
X_train, y_train = train_dataset.data / 255, train_dataset.targets
X_test, y_test = train_dataset.data / 255, train_dataset.targets
#자료의 구조를 보면 6000,28,28의 자료 형태를 갖는다. 하습 시키기 위해 
#2차원 배열으로 변환해야 하는데 28*28=784를 속성의 개수로 만들어준다
#view(-1,784)는 1차원은 알아서 하고 2차원을 784로 하라는 의미이다. 
X_train, X_test = X_train.view(-1, 784), X_test.view(-1, 784)

train_dset = TensorDataset(X_train, y_train)
test_dset= TensorDataset(X_test, y_test)

#컴퓨터의 한계 때문에 각각의 학습데이터를 배치 단위로 나눠서 연산을 한다.
train_loader = DataLoader(train_dset,  batch_size=30, shuffle=True)
test_loader = DataLoader(test_dset,  batch_size=30, shuffle=False)

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

device = 'cpu' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


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
    print('epoch: {}, loss: {}, acc;{}'.format(epoch, loss, acc))  
