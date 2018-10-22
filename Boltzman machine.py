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


#converting the rating into binary ratings 1(Liked) or 0 (Not like) or -1 (not rated yet)
training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set>=3] = 1
training_set[test_set==0] = -1
training_set[test_set==1] = 0
training_set[test_set==2] = 0
training_set[test_set>=3] = 1

#creating the architecture of the Neural network
class RBM():
	def _init_(self,nv,nh):
		self.W = torch.randn(nh,nv)
		self.a = torch.randn(1,nh)
		self.b = torch.randn(1,nv)
	def sample_h(self,x):
		wx = torch.mm(x,self.W.t())
		activation = wx+self.a.expand_as(wx)
		p_h_given_v = torch.sigmoid(activation)
		return p_h_given_v, torch.bernoulli(p_h_given_v)
	def sample_v(self,y):
		wy = torch.mm(y,self.W)
		activation = wy+self.b.expand_as(wy)
		p_v_given_h = torch.sigmoid(activation)
		return p_v_given_h,torch.bernoulli(p_v_given_h)
	def train(self,v0,vk,ph0,phk):
		self.W += torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)
		self.b += torch.sum((v0-vk),0)
		self.a += torch.sum((ph0-phk),0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh)

#Training the RBM
nb_epoch = 10
for epoch in range(1,nb_epoch+1):
	train_loss = 0
	s = 0.
	for id_user in range(0,nb_users-batch_size,batch_size):
		vk = training_set[id_user:id_user+batch_size]
		v0 = training_set[id_user:id_user+batch_size]
		ph0,_ = rbm.sample_h(v0)
		for k in range(10):
			_,hk = rmb.sample_h(vk)
			_,vk = rmb.sample_v(hk)
			vk[v0<0] = v0[v0<0]
		phk,_ = rbm.sample_h(vk)
		rbm.train(v0,vk,ph0,phk)
		train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
		s += 1.
	print ('epoch:'+str(epoch)+'loss:'+str(train_loss/s))

#Test the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
	v = training_set[id_user:id_user+1]  #input
	vt = test_set[id_user:id_user+1] #target
	if len(vt[vt>=0]) > 0:
		_,h = rmb.sample_h(v)
		_,v = rmb.sample_v(h)
	test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
	s += 1.
print ('test loss:'+str(test_loss/s))


