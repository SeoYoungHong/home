from sklearn import datasets
from torch.serialization import load
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

dataset = datasets.load_breast_cancer()

X, y = dataset['data'], dataset['target']
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(-1)

X = (X-torch.mean(X))/torch.std(X)

#데이터 묶음을 만들고 한번에 256개씩 불러올 수 있도록 한다.
#shuffle==True로 해서 매 에포크마다 섞이도록한다.
dset = TensorDataset(X, y)
loader = DataLoader(dset, batch_size = 256, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear1 = nn.Linear(num_features, 4)#입력층->은닉층
        self.relu = nn.ReLU()#은닉층
        self.linear2 = nn.Linear(4, 1)#은닉층->출력층
        self.sigmoid = nn.Sigmoid()#출력층
    def forward(self, X):
        out = self.linear1(X)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

model = NeuralNetwork(30)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#하나의 에포크를 베치단위로 나눠서 시행을 시키다.
def train(model, criterion, optimizer, loader):
    epoch_loss = 0 #현재 에포크의 오차를 저장할 변수
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(X_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    #오차를 다 합친 후에 나눠진 배치의 개수만큼 나눠 평균을 게산한다.
    return epoch_loss / len(loader)

n_epoch = 100
for epoch in range(0,n_epoch+1):
    loss = train(model, criterion, optimizer, loader)
    if epoch % 10 == 0:
        print('epoch: {}, loss: {}'.format(epoch, loss))

y_predicted1 = (model(X) >= 0.5).float()
score = (y_predicted1==y).float().mean()
print('accuracy1: {:.2f}'.format(score))

torch.save(model.state_dict(), './trained_model.pt')
load_model = NeuralNetwork(30)

load_model.load_state_dict(torch.load('./trained_model.pt'))
y_predicted2 = (load_model(X) >= 0.5).float()
score = (y_predicted2==y).float().mean()
print('accuracy2: {:.2f}'.format(score))