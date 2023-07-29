import torch

# create a batch of 3 states and 2 actions per state
batch_size = 3
state_size = 4
action_size = 2
batch_state = torch.randn(batch_size, state_size)
batch_action = torch.randint(action_size, size=(batch_size,))

# create a simple model that takes a state and outputs a Q-value for each action
model = torch.nn.Linear(state_size, action_size)

# compute the Q-values for the actions taken in each state
outputs = model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

# print the resulting tensor
print(outputs.shape)
