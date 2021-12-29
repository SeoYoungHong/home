from sklearn import datasets
import torch
import torch.nn as nn #다양한 알고리즘이 있음
import torch.optim as optim #옵티마이저(에러를 최적화하는 방법)
from ModelClass import LinearRegretion

dataset = datasets.load_boston()
X, y = dataset['data'], dataset['target']

X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(-1) #불러온 데이터는 1차원 배열인데 파이토치는 2차원배열을 가지고 연산을 하므로 2차원 배열으로 만들어준다 [n,] -> [n,1]

X = (X-torch.mean(X)) / torch.std(X) #선형회귀는 범위에 민감하므로 표준화를 시켜준다
model = LinearRegretion(13) #13개의 입력을 받고 하나의 출력을 내는 선형회귀를 만든다.

criterion = nn.MSELoss() #오차를 계산할 방법을 결정한다.
optimizer = optim.SGD(model.parameters(), lr=0.01) #모델의 변수와 학습율을 알려주고 모델의 변수를 수정하는 방법을 결정하는 것이다. 여기서는 경사하강법으로 결정했다.

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
