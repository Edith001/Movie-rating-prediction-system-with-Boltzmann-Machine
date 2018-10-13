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
