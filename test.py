import torch
print('hi')
device = 'cuda' 
if torch.cuda.is_available():
    print('cuda') 
else :
    print('cpu')
print(torch.__version__)
