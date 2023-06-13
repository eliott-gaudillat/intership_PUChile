import numpy as np

class ReplayBuffer():

	def __init__(self,max_size, input_shape,n_actions):
		self.memory_size=max_size
		self.mem_cpt=0
		self.state_memory = np.zeros((self.memory_size, *input_shape))
		self.new_state_memory= np.zeros((self.memory_size, *input_shape))
		self.action_memory= np.zeros((self.memory_size, n_actions))
		self.reward_memory= np.zeros(self.memory_size)
		self.terminated_memory= np.zeros(self.memory_size, dtype=np.bool_)
		
	
	def save_transition(self,state,action,reward,state_,done):
		#define index
		index = self.mem_cpt % self.memory_size
		#save transition in buffer from position index
		self.state_memory[index]=state
		self.new_state_memory[index] = state_
		self.action_memory[index]=action
		self.reward_memory[index]=reward
		self.terminated_memory[index]=done
		#incrementation cpt
		self.mem_cpt +=1
		
	def sample_buffer(self,batch_size):
		#memory size used
		max_mem=min(self.mem_cpt,self.memory_size)
		#take batch_size random number between 0 and max_mem
		batch=np.random.choice(max_mem,batch_size)
		# take the element in the buffer located in the bacth position
		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.terminated_memory[batch]
		return states,actions,rewards,states_,dones
