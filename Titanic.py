from numpy.core.numeric import NaN
import torch
import numpy as np
import torch.nn as nn 
import pandas as pd
import math

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from torch.utils.data import Dataset, DataLoader

class neuralnet(nn.Module):
	def __init__(self, input_dim=7, output_dim=1, hidden_dim=64, num_layers=0, p_dropout=0.0, device='cpu'):
		super().__init__()
		self.layers = nn.Sequential()
		self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

		prev_size = input_dim
		this_size = hidden_dim
		for i in range(num_layers):
			self.layers.add_module('layer{}'.format(i), nn.Linear(prev_size, this_size))
			self.layers.add_module('activation{}'.format(i), nn.ReLU())
			self.layers.add_module('DropOut{}'.format(i), nn.Dropout(p=p_dropout))
			prev_size = hidden_dim
		self.layers.add_module('outputLayer', nn.Linear(prev_size, output_dim))
		self.layers.add_module('1234', nn.Sigmoid())
		self.to(self.device)

	def forward(self, input):
		return self.layers(input)

def testing(model, dataset, targets,batch):
	DATA = torch.tensor(dataset).type(torch.float32)
	TARGETS = torch.tensor(targets).type(torch.float32)
	i = 0
	a = len(TARGETS)/batch

	predict = []
	real = []
	for i in tqdm(range(math.ceil(a)), leave=True):
		data = DATA[batch*i:batch*(i+1),:]
		target = TARGETS[batch*i:batch*(i+1)]
		data = data.view([-1, 7]).to(model.device)
		target = target.squeeze().to(model.device)

		predict.extend(np.where(model(data).cpu().data.numpy() > 0.5, 1, 0).squeeze())
		real.extend(target.cpu().numpy().tolist())

	return np.mean(np.array(predict) == np.array(real))


def training(model, dataset, targets, loss_func, optimizer, epochs, batch):
	for epoch in tqdm(range(epochs), leave=False):
		DATA = torch.tensor(dataset).type(torch.float32)
		TARGETS = torch.tensor(targets).type(torch.float32)
		i = 0
		a = len(TARGETS)/batch

		for i in tqdm(range(math.ceil(a)), leave=False):
			data = DATA[batch*i:batch*(i+1),:]
			target = TARGETS[batch*i:batch*(i+1)]
			optimizer.zero_grad()
			data = data.view([-1, 7]).to(model.device)
			target = target.squeeze().to(model.device)

			output = torch.squeeze(model(data))
			loss = loss_func(output, target)
			loss.backward()
			optimizer.step()

def predict(data):
	i = 0

def pr(x):
	if x == 'S': return (1)
	elif x == 'C': return (2)
	elif x == 'Q': return (3)
	else: return (0)

def edit(DATA, if_train):
	for i in ['PassengerId','Ticket','Name','Cabin']: DATA.pop(i)
	DATA['Sex'] = (DATA['Sex']=='male').astype(int)
	if if_train: TARGET = DATA.pop('Survived').astype(int)
	DATA['Age'] = DATA['Age'].fillna(DATA['Age'].mean())
	DATA['Embarked'] = DATA['Embarked'].fillna(0)
	DATA['Fare'] = DATA['Fare'].fillna(0)

	sc = MinMaxScaler()
	DATA['Embarked'] = DATA['Embarked'].apply(pr)
	DATA[['Age','Fare', 'SibSp', 'Parch']] = sc.fit_transform((DATA[['Age','Fare', 'SibSp', 'Parch']]).to_numpy(dtype=float).reshape(-1,4))
	if if_train: return(np.array(DATA, dtype=float), np.array(TARGET, dtype=float))
	else: return np.array(DATA, dtype=float)

a = pd.read_csv("./VSCODE PYTHON/Kaggle/Titanic/train.csv")
b = pd.read_csv("./VSCODE PYTHON/Kaggle/Titanic/test.csv")
DATA_TR, TARGET_TR = edit(a,1)
DATA_TEST = edit(b,0)

model = neuralnet()
training(model, DATA_TR, TARGET_TR, nn.BCELoss(),
	     torch.optim.Adam(model.parameters(), lr= 0.001, weight_decay=0.001),
		 epochs= 50,
		 batch = 9)
result = testing(model, DATA_TR, TARGET_TR, 9)
print(result)
