
# importing libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights , std=1.0):
	out = torch.randn(weights.size())
	
	out_squared_sum = out.pow(2).sum(1, keepdim=True)

	out_expanded = out_squared_sum.expand_as(out)

	out *= std / torch.sqrt(out_expanded)

	return out

def weights_init(m):

	class_name = m.__class__.__name__

	if class_name.find('Conv') !=-1:
		
		weight_shape = list(m.weight.data.size())

		fan_in = np.prod(weight_shape[1:4]) # dim1 * 2 *3

		fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * 2 * 3

		w_bound = np.sqrt(6. / fan_in + fan_out)

		m.weight.data.uniform_(-w_bound , w_bound)

		m.bias.data.fill_(0)

	elif class_name.find('Linear') !=-1:
		weight_shape = list(m.weight.data.size())
		
		fan_in = weight_shape[1]

		fan_out = weight_shape[0]

		w_bound = np.sqrt(6. / fan_in + fan_out)

		m.weight.data.uniform_(-w_bound , w_bound)

		m.bias.data.fill_(0)


# Making the A3C brain

class ActorCritic(torch.nn.Module):
	"""docstring for ActorCritic"""
	def __init__(self, num_inputs , action_space):
		super(ActorCritic, self).__init__()

		self.num_inputs= num_inputs

		self.action_space = action_space

		self.conv1 = nn.Conv2d(num_inputs , 32,3,stride=2,padding=1)

		self.conv2 = nn.Conv2d(32 , 32,3,stride=2,padding=1)

		self.conv3 = nn.Conv2d(32 , 32,3,stride=2,padding=1)

		self.conv4 = nn.Conv2d(32 , 32,3,stride=2,padding=1)

		self.lstm = nn.LSTMCell(32 * 3 * 3 , 256)

		num_outputs = action_space.n

		self.critic_linear = nn.Linear(256 , 1) # output = V(s) shared among actors

		self.actor_linear = nn.Linear(256,num_outputs) # output = number of Q-values Q(S,A)

		self.apply(weights_init) # initialize some random weights

		self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data,0.01)
		
		self.actor_linear.bias.data.fill_(0)

		self.critic_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data,0.01)
		
		self.critic_linear.bias.data.fill_(0)

		self.lstm.bias_ih.data.fill_(0)

		self.lstm.bias_hh.data.fill_(0)

		self.train()

	def forward(self , inputs):

		inputs , (hx,cx) = inputs

		x = F.elu(self.conv1(inputs))

		x = F.elu(self.conv2(inputs))

		x = F.elu(self.conv3(inputs))

		x = F.elu(self.conv4(inputs))

		x = x.view(-1,32 * 3 * 3)

		(hx,cx) = self.lstm(x,(hx,cx))

		x = hx

		return self.critic_linear(x) , self.actor_linear(x) , (hx,cx)










