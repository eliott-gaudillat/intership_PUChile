import math
import random
import gym
import numpy as np
import mujoco_py
import glfw
import time
import os
from jacobienne import MGIv2
from jacobienne import SmallestSignedAngleBetween









# Para visualizar que es lo que hace el agente
glfw.init()
test_steps = 1000
env = gym.make("Reacher-v4", render_mode="human") # brazo 2 DOF



Kp=0.04
Kd=0.02

episode=0
num_episode=15
while episode<num_episode:
	episode+=1
	observation, info = env.reset()
	# recuperation des X et Y de la cible
	x_star=observation[4]
	y_star=observation[5]
	#calcul des q1,q2 desirée
	q1_star,q2_star=MGIv2(x_star,y_star)

	# calcul des q1 et q2 actuel
	cosq1=observation[0]
	cosq2=observation[1]
	sinq1=observation[2]
	sinq2=observation[3]

	q1=np.arctan2(sinq1,cosq1)
	q2=np.arctan2(sinq2,cosq2)

	#calcul de l'erreur sur chaque articulation
	error_q1=SmallestSignedAngleBetween(q1_star,q1)
	error_q2=SmallestSignedAngleBetween(q2_star,q2)


	while((abs(error_q1)>1e-4 and abs(error_q2)>1e-4)or terminated):
		# calcul des q1 et q2 actuel
		cosq1=observation[0]
		cosq2=observation[1]
		sinq1=observation[2]
		sinq2=observation[3]

		q1=np.arctan2(sinq1,cosq1)
		q2=np.arctan2(sinq2,cosq2)
	
		error_q1_point=-observation[6]
		error_q2_point=-observation[7]
		#calcul de l'erreur sur chaque articulation
		error_q1=q1_star-q1
		error_q2=q2_star-q2
		#definir nouvelle valeur/controle a appliquer au torque
		U1=Kp*error_q1+Kd*error_q1_point
		U2=Kp*error_q2+Kd*error_q2_point
		action= [U1 ,U2]
		observation, reward, terminated, truncated, info = env.step(action)
		time.sleep(0.1)
     
  
	print(" valeurs final :episode n:",episode," error_q1 :",error_q1," error_q2 :", error_q2,"terminated:",terminated)
	
glfw.terminate()
env.close()


