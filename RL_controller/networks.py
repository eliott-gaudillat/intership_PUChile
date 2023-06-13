import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
	def __init__(self,beta,obs_dims,actions_dims,fc1_dims=256,fc2_dims=256,name='critic',temp_dir='tmp/sac'):
		super(CriticNetwork,self).__init__()
		self.obs_dims=obs_dims
		self.actions_dims=actions_dims
		self.fc1_dims=fc1_dims
		self.fc2_dims=fc2_dims
		self.name=name
		self.temp_dir=temp_dir
		self.temp_file=os.path.join(self.temp_dir,name+'_sac')
		self.temp_file_end=os.path.join(self.temp_dir,name+'_sac_end')
		
		#define neural network
		#entry dimension obs+act because critic value the pair obs/act
		self.fc1=nn.Linear(self.obs_dims[0]+self.actions_dims,self.fc1_dims)
		self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
		self.q=nn.Linear(self.fc2_dims,1)
		
		#Optimizer
		#lr means learning rate
		self.optimizer=optim.Adam(self.parameters(), lr=beta)
		#device
		self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)
	
	#define entry of each layer
	def forward(self,state,action):
		action_value=self.fc1(T.cat([state,action],dim=1))
		action_value=F.relu(action_value)
		action_value=self.fc2(action_value)
		action_value=F.relu(action_value)
		q=self.q(action_value)
		
		return q
		
		
	def save_checkpoint(self,end=False):
		if not end:
			T.save(self.state_dict(), self.temp_file)
		else:
			T.save(self.state_dict(),self.temp_file_end)
		
	def load_checkpoint(self):
		self.load_state_dict(T.load(self.temp_file))
        	
        	
class ValueNetwork(nn.Module):
	def __init__(self,beta,obs_dims,fc1_dims=256,fc2_dims=256,name='value',temp_dir='tmp/sac'):
		super(ValueNetwork,self).__init__()
		self.obs_dims=obs_dims
		self.fc1_dims=fc1_dims
		self.fc2_dims=fc2_dims
		self.name=name
		self.temp_dir=temp_dir
		self.temp_file=os.path.join(self.temp_dir,name+'_sac')
		self.temp_file_end=os.path.join(self.temp_dir,name+'_sac_end')
		#define neural network
		self.fc1=nn.Linear(*self.obs_dims,self.fc1_dims)
		self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
		self.v=nn.Linear(self.fc2_dims,1)
		
		#Optimizer
		#lr means learning rate
		self.optimizer=optim.Adam(self.parameters(), lr=beta)
		#device
		self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)
		
	def forward(self,state):
		state_value=self.fc1(state)
		state_value=F.relu(state_value)
		state_value=self.fc2(state_value)
		state_value=F.relu(state_value)
		v=self.v(state_value)
		
		return v
		
		
	def save_checkpoint(self,end=False):
		if not end:
			T.save(self.state_dict(), self.temp_file)
		else:
			T.save(self.state_dict(),self.temp_file_end)
		
	def load_checkpoint(self):
    		self.load_state_dict(T.load(self.temp_file))
        	

class ActorNetwork(nn.Module):
	def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        	super(ActorNetwork, self).__init__()
        	self.input_dims = input_dims
        	self.fc1_dims = fc1_dims
        	self.fc2_dims = fc2_dims
        	self.n_actions = n_actions
        	self.name = name
        	self.temp_dir = chkpt_dir
        	self.temp_file = os.path.join(self.temp_dir, name+'_sac')
        	self.temp_file_end=os.path.join(self.temp_dir,name+'_sac_end')
        	self.max_action = max_action
        	self.reparam_noise = 1e-6
        	self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        	self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        	self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        	self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        	self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        	self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        	self.to(self.device)
        	
	def forward(self, state):
	        prob = self.fc1(state)
	        prob = F.relu(prob)
        	prob = self.fc2(prob)
        	prob = F.relu(prob)

        	mu = self.mu(prob)
        	sigma = self.sigma(prob)

        	sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        	return mu, sigma
        	
        	
	def sample_normal(self, state, reparameterize=True):
		mu, sigma = self.forward(state)
		probabilities = Normal(mu, sigma)
		if reparameterize:
			actions = probabilities.rsample()
		else:
			actions = probabilities.sample()
		action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
		log_probs = probabilities.log_prob(actions)
		log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
		log_probs = log_probs.sum(1, keepdim=True)
		
		return action, log_probs
        	
	def save_checkpoint(self,end=False):
		if not end:
			T.save(self.state_dict(), self.temp_file)
		else:
			T.save(self.state_dict(),self.temp_file_end)
		
	def load_checkpoint(self):
		self.load_state_dict(T.load(self.temp_file))
	
		
		
	
