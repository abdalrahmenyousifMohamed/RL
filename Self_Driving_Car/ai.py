import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):

	def __init__(self,input_size,nb_actions):
		super(Network,self).__init__()
		self.input_size = input_size
		self.nb_actions = nb_actions
		self.fc1 = nn.Linear(input_size ,60)
		self.fc2 = nn.Linear(60,30)
		self.layer_out = nn.Linear(30,nb_actions)

		self.dropout = nn.Dropout(p=0.1)
		#self.norm1 = nn.BatchNorm1d(2)
		# self.norm2 = nn.BatchNorm1d(30)

	def forward(self,state):
		x = F.relu(self.fc1(state))
		#x = self.norm1(x)
		x = F.relu(self.fc2(x))
		# x = self.norm2(x)
		x = self.dropout(x)
		x = self.layer_out(x)
		q_values = x

		return q_values

#implementing Experiance Replay

class ReplayMemory(object):

	def __init__(self,capacity):

		self.capacity = capacity # ensure memory list always contain 100 events and never more

		self.memory = [] # will contain the last 100 transactions as a maximum numbers


	def push(self,event):

		self.memory.append(event)

		if len(self.memory) > self.capacity:

			del memory[0]

	def sample(self,batch_size):

		sample = zip(*random.sample(self.memory , batch_size))

		return map(lambda x : Variable(torch.cat(x,0)) , sample)


# Implementing  Deep Q Learning

class Dqn():
	def __init__(self,input_size,nb_actions,gamma):
		self.gamma = gamma
		self.reward_window = [] # mean of the rewards over time
		self.model = Network(input_size,nb_actions)
		self.memory = ReplayMemory(100000)
		self.optimizer = optim.Adam(self.model.parameters() , lr=0.001)
		self.last_state = torch.Tensor(input_size).unsqueeze(0) # batch_size and input_tensor
		self.last_action = 0
		self.last_reward = 0
	def select_action(self,state):
		with torch.no_grad():
			probs = F.softmax(self.model(torch.Tensor(state))*60)

		action = probs.multinomial(1)

		return action.data[0]

	def learn(self , batch_state,batch_next_state,batch_reward , batch_action):

		outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1) #  Q(StB, atB),Q(s,a) =V(s) + A(a) Index tensor must have the same number of dimensions as input tensor

		next_outputs = self.model(batch_next_state).detach().max(1)[0] # maximum of q values of next states according to all actions represented by iDX 1

		target = self.gamma * next_outputs + batch_reward # Rts + ymax(Q(a, StB+1)) # Q-Target = r + γQ(s’,argmax(Q(s’,a,ϴ),ϴ’))


		temporal_difference_loss = F.smooth_l1_loss(outputs,target)

		self.optimizer.zero_grad()

		temporal_difference_loss.backward()

		self.optimizer.step() 

	def update(self,reward , new_signal):

		new_state = torch.Tensor(new_signal).float().unsqueeze(0)

		self.memory.push((self.last_state,new_state,torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward])))

		action = self.select_action(new_state)

		if len(self.memory.memory) > 100:

			batch_state,batch_next_state,batch_action, batch_reward = self.memory.sample(100)

			self.learn(batch_state,batch_next_state,batch_reward , batch_action)

		self.last_action = action

		self.last_state = new_state

		self.last_reward = reward

		self.reward_window.append(reward)

		if len(self.reward_window) > 1000:

			del self.reward_window[0]

		return action

	def score(self):

		return sum(self.reward_window) / (len(self.reward_window)+1)


	def save(self):

		torch.save({'state_dict' : self.model.state_dict() ,

		 'optimizer' :self.optimizer.state_dict() },'last_brain.pth'
		 )

	def load(self):

		if os.path.isfile('last_brain.pth'):

			print('=> loading checkpoint...')

			checkpoint = torch.load('last_brain.pth')

			self.model.load_state_dict(checkpoint['state_dict'])

			self.optimizer.load_state_dict(checkpoint['optimizer'])

			print('done ! ')


		else:

			print('no checkpoint found ...')








