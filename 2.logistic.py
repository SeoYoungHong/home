from sklearn import datasets
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import torch.optim as optim

dataset = datasets.load_breast_cancer
X, y = dataset['data'], dataset['targert']

X = torch.FloatTensor(X)
y = torch.FloatTensor(y).view(-1,1)

X = (X-torch.mean)/torch.std(X)

model = nn.Sequential(#여러개의 신경망을 엮어서 사용할 수 있다.
    nn.Linear(30,1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def train(model, criterion, X, y):
  optimizer.zero_grad()
  hypothesis = model(X)
  loss = criterion(hypothesis, y) #nn라이브러리를 가지고 있다.
  loss.backward() #오차를 가지고 계산할 수 있다.
  optimizer.step() #계산된 기울기를 가지고 결정된 수정 방식에 맞게 변수를 수정한다.
  return loss.item()

n_epochs = 100
for epoch in range(1, n_epochs):
  loss = train(model, criterion, X, y)
  if epoch % 10 == 0:
    print('epoch: {}, loss: {}'.format(epoch, loss))