from torch._C import device
from torch.nn.modules.activation import ReLU
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import time
start = time.time()

path = './'
train_dataset = datasets.MNIST(path, train=True, download=True)
test_dataset = datasets.MNIST(path, train=False, download=True)

X_train, y_train = train_dataset.data / 255, train_dataset.targets
X_test, y_test = test_dataset.data / 255, test_dataset.targets

print('학습 세트 입력 데이터 :',X_train.shape)
print('학습 세트 타깃:', y_train.shape)
print('테스트 세트 입력 데이터:', X_test.shape)
print('테스트 세트 타깃:', y_test.shape)

X_train, X_test = X_train.unsqueeze(1), X_test.unsqueeze(1)

train_dset = TensorDataset(X_train, y_train)
test_dset= TensorDataset(X_test, y_test)

print('학습 세트 입력데이터:', X_train.shape)
print('테스트 세트 입력데이터:', X_test.shape)

train_loader = DataLoader(train_dset,  batch_size=30, shuffle=True)
test_loader = DataLoader(test_dset,  batch_size=30, shuffle=False)

class CNN(nn.Module):
    def __init__(self) :
        super().__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.5)
        )
        self.hidden_layer3 = nn.Linear(128*5*5, 128)
        self.output_layer = nn.Linear(128, 10)
    def forward(self, X):
        out = self.hidden_layer1(X)
        out = self.hidden_layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.hidden_layer3(out)
        out = self.output_layer(out)
        return out

device = 'cpu' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

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

def evaluate(model, criterion, loader):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            hypothesis = model(X_batch)
            loss = criterion(hypothesis, y_batch)
            y_predicted2 = torch.argmax(hypothesis, 1)
            acc = (y_predicted2==y_batch).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    #오차를 다 합친 후에 나눠진 배치의 개수만큼 나눠 평균을 게산한다.
    return epoch_loss / len(loader), epoch_acc / len(loader)

n_epoch = 20
for epoch in range(0,n_epoch+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    loss2, acc2 = evaluate(model, criterion, test_loader)
    print('epoch: {}, loss: {}, acc;{}'.format(epoch, loss, acc))
    print('epoch: {}, loss2: {}, acc2;{}'.format(epoch, loss2, acc2))   
    print(time.time()-start) 
