
import gym
#import gym.envs.registration
import numpy as np
from sac import Agent
from utils import plot_learning_curve
from gym import wrappers
import math
import random
import mujoco_py
import glfw
import time
import os
from jacobienne import MGIv2
from jacobienne import SmallestSignedAngleBetween




def SmallestSignedAngleBetween(theta1, theta2):     # Para que el angulo se mantenga en el rango [-pi, pi]
    a = (theta1 - theta2) % (2.*np.pi)
    b = (theta2 - theta1) % (2.*np.pi)
    return -a if a < b else b
 





if __name__ == '__main__':
	glfw.init()
	env = gym.make("Reacher-v4", max_episode_steps=130,render_mode="human")
	
	agent=Agent(input_dims=env.observation_space.shape,env=env,n_actions=env.action_space.shape[0])
	n_epochs=6
	n_games=250
	Kdist=1
	Kctrl=1
	filename='reacher.png'
	
	figure_file='plots/'+filename
	
	best_score=env.reward_range[0]
	score_history=[]
	#if you want or not load a pre-trained model
	
	time_game=0
	tps1 = time.time()
	load_checkpoint=True
	if load_checkpoint:
		agent.load_models()

	for j in range(n_epochs):
		for i in range (n_games):
			observation,_=env.reset(random=j)
			done=False
			score =0
			error_q1=1
			error_q2=1
			q1_star,q2_star=MGIv2(observation[4],observation[5])
			while( not done): 
				action=agent.choose_action(observation)
				# for Reacher reward it a sum of 2 differents reward
				observation_, reward, terminated, truncated, info = env.step(action)
				reward=Kdist*info.get('reward_dist')+Kctrl*info.get('reward_ctrl')
				cosq1,cosq2,sinq1,sinq2,x_star,y_star,q1_point,q2_point,x_error,y_error=observation[0:10]
				q1=np.arctan2(sinq1,cosq1)
				q2=np.arctan2(sinq2,cosq2)
				#calcul de l'erreur sur chaque articulation
				error_q1=q1_star-q1
				error_q2=q2_star-q2
				done=terminated or truncated or (abs(error_q1)<1e-4 and abs(error_q2)<1e-4)
				score+=reward
				agent.remember(observation,action,reward,observation_,done)
				if not load_checkpoint:
					agent.learn()
				observation=observation_
				time.sleep(0.1)
			score_history.append(score)
			avg_score=np.mean(score_history[-100:])
			if((i+j)==0):
				tps2 = time.time()
				time_game=tps2 - tps1	
					
			if(i%50==0):
				t_total_sec=(n_epochs-j-1)*(n_games-i-1)*time_game
				t_total_minute=t_total_sec/60
				t_minute_left=t_total_minute%60
				t_total_heure=t_total_minute//60
				print("temps estime:",t_total_heure,"h",t_minute_left,"min")
				print("epochs numero:",j,"game numero :",i,"score:",score)	
		
			if avg_score>best_score:
				best_score=avg_score
				if not load_checkpoint:
					agent.save_models()
			if i==0:
				best_score=avg_score
				if not load_checkpoint:
					agent.save_models()
					
			
		print("epochs numero :" ,j,"avg_score:",avg_score)		
		if not load_checkpoint:
			x=[i+1 for i in range(n_games*(j+1))]
			plot_learning_curve(x,score_history,figure_file)
	if not load_checkpoint:
		agent.save_models(end=True)
