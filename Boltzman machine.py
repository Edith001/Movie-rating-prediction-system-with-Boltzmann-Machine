import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importing the dataset
movies = pd.read_csv('ml-lm/movies.dat',sep='::',header = None,engine = 'python',encoding = 'latin-1')
users = pd.read_csv('ml-lm/users.dat',sep='::',header = None,engine = 'python',encoding = 'latin-1')
rating = pd.read_csv('ml-lm/ratings.dat',sep='::',header = None,engine = 'python',encoding = 'latin-1')

#preparing the training set and test set
training_set = pd.read_csv('ml-100k/ul.base',delimiter = '\t')
training_set = np.array(training_set,dtype = 'int')
test_set = pd.read_csv('ml-100k/ul.test',delimiter = '\t')
test_set = np.array(test_set,dtype='int')

#put all users and movies into an array with all user id and movie ids with the value of rating,
# If the user haven't rate the movie, the value is 0

#Getting the number of total users and total movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Converting the data into an array with users in rows and movies in columns
# the array is a list of list, which is the data structure needed to transfer into torch sensor
def convert(data):
	new_data = []
	for id_users in range(1,nb_users+1):
		id_movies = data[:,1][data[:,0] == id_users]
		id_ratings = data[:,2][data[:,0] == id_users]
		ratings = np.zeros(nb_movies)
		ratings[id_movies-1] = id_ratings
		new_data.append(lists(ratings))
	return new_data
training_set = convert(training_set)
test_set = convert(test_set)

#convert the data into torch tensors
training_set = torch.FloatTensor(trainig_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
	def _int_(self,):
		super(SAE,self)._init_()
		self.fc1 = nn.Linear(nb_movies,20)
		self.fc2 = nn.Linear(20,10)
		self.fc3 = nn.Linear(10,20)
		self.fc4 = nn.Linear(20,nb_movies)
		self.activation = nn.Sigmoid()
	def forward(self,x):
		self.activation(self.fc1(x))
		self.activation(self.fc2(x))
		self.activation(self.fc3(x))
		x = self.fc4(x)
		return x
sae = SAE()
criterion = MSELoss()
optimizer = optim.RMSprop(sae.parameters(),lr = 0.01,weight_decay = 0.5)

#Training the SAE
nb_epoch = 200
for epoch in range(1,nb_epoch+1):
	train_loss = 0
	a = 0.
	for id_user in range(nb_users):
		input = Variable(training_set[id_user]).unsqueeze(0)
		target = input.clone()
		if torch.sum(target.data > 0) > 0:
			output = sae(input)
			target.require_grad = False
			output[target==0] = 0
			loss = criterion(output,target)
			mean_corrector = nb_movies/float(torch.sum(target.data>0)+1e-10)
			loss.backward()
			train_loss += np.sqrt(loss.data[0]*mean_corrector)
			s += 1.
			optimizer.step()
	print ('epoch:'+str(epoch)+'loss:'+str(train_loss/s))













